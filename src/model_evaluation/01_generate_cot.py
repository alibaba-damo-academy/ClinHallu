import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
for candidate in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from common import (
    CoTAPIWrapper,
    REPLACE_EXPERIMENTS,
    align_samples,
    build_options_text,
    create_model_cot_wrapper,
    encode_image_to_base64,
    extract_step_text,
    format_question_with_options,
    get_sample_identifier,
    get_generation_request_config,
    get_mime_type,
    get_replace_generation_file,
    load_completed_sample_ids,
    load_config,
    load_input_data,
    load_jsonl,
    load_questions_from_gold_cot,
    parse_cot,
    resolve_image_paths,
    resolve_paths,
    validate_cot,
)
from prompts.cot_generation import (
    CONTINUE_FROM_KR_PROMPT,
    CONTINUE_FROM_VR_KR_PROMPT,
    CONTINUE_FROM_VR_PROMPT,
    COT_SYSTEM_PROMPT,
    COT_USER_PROMPT_TEXT_ONLY,
    COT_USER_PROMPT_WITH_IMAGE,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _require_sample_identifier(sample: Dict[str, Any]) -> str:
    sample_id = get_sample_identifier(sample)
    if sample_id is None:
        raise ValueError(
            "Sample is missing source_id/original_index and cannot be tracked by ID: "
            f"question={sample.get('question', '')!r}"
        )
    return sample_id


@dataclass
class GenerationJob:
    output_sample: Dict[str, Any]
    image_sample: Dict[str, Any]
    request_question: str
    mode: str
    prompt_variant: str
    options_text: str = ""
    custom_prompt: Optional[str] = None
    prefix_steps: List[Tuple[str, str]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def _assemble_raw_cot(prefix_steps: Sequence[Tuple[str, str]], raw_response: str) -> str:
    prefix_blocks = [
        f"[{step_name}]\n{step_text.strip()}"
        for step_name, step_text in prefix_steps
        if step_text and step_text.strip()
    ]
    raw_response = raw_response.strip()
    if not prefix_blocks:
        return raw_response
    prefix = "\n\n".join(prefix_blocks)
    if not raw_response:
        return prefix
    return f"{prefix}\n\n{raw_response}"


def _prepare_replace_output_sample(
    gold_sample: Dict[str, Any],
    experiment: str,
) -> Dict[str, Any]:
    output_sample = {
        key: value
        for key, value in gold_sample.items()
        if key not in ("gold_cot", "gold_cot_meta", "cot_valid")
    }
    output_sample["experiment"] = experiment
    return output_sample


def _build_replace_job(
    gold_sample: Dict[str, Any],
    model_sample: Dict[str, Any],
    experiment: str,
) -> GenerationJob:
    gold_cot = gold_sample.get("gold_cot", {})
    model_cot = model_sample.get("cot", {})
    question_text = format_question_with_options(gold_sample)
    output_sample = _prepare_replace_output_sample(gold_sample, experiment)

    if experiment == "replace_vr":
        gold_vr_text = extract_step_text(gold_cot, "visual_recognition")
        prompt = CONTINUE_FROM_VR_PROMPT.format(
            question=question_text,
            visual_recognition=gold_vr_text,
        )
        prefix_steps = [("Visual Recognition", gold_vr_text)]
        meta = {"replaced_steps": ["visual_recognition"]}
    elif experiment == "replace_kr":
        model_vr_text = extract_step_text(model_cot, "visual_recognition")
        gold_kr_text = extract_step_text(gold_cot, "knowledge_recall")
        prompt = CONTINUE_FROM_KR_PROMPT.format(
            question=question_text,
            visual_recognition=model_vr_text,
            knowledge_recall=gold_kr_text,
        )
        prefix_steps = [
            ("Visual Recognition", model_vr_text),
            ("Knowledge Recall", gold_kr_text),
        ]
        meta = {
            "replaced_steps": ["knowledge_recall"],
            "preserved_steps": ["visual_recognition"],
        }
    elif experiment == "replace_vr_kr":
        gold_vr_text = extract_step_text(gold_cot, "visual_recognition")
        gold_kr_text = extract_step_text(gold_cot, "knowledge_recall")
        prompt = CONTINUE_FROM_VR_KR_PROMPT.format(
            question=question_text,
            visual_recognition=gold_vr_text,
            knowledge_recall=gold_kr_text,
        )
        prefix_steps = [
            ("Visual Recognition", gold_vr_text),
            ("Knowledge Recall", gold_kr_text),
        ]
        meta = {
            "replaced_steps": ["visual_recognition", "knowledge_recall"],
        }
    else:
        raise ValueError(f"Unsupported replace experiment: {experiment}")

    return GenerationJob(
        output_sample=output_sample,
        image_sample=gold_sample,
        request_question=gold_sample.get("question", ""),
        mode=experiment,
        prompt_variant=experiment,
        custom_prompt=prompt,
        prefix_steps=prefix_steps,
        meta=meta,
    )


def build_replace_jobs(
    aligned_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    experiment: str,
) -> List[GenerationJob]:
    return [
        _build_replace_job(gold_sample, model_sample, experiment)
        for gold_sample, model_sample in aligned_pairs
    ]


def build_standard_jobs(samples: List[Dict[str, Any]], mode: str) -> List[GenerationJob]:
    return [
        GenerationJob(
            output_sample=dict(sample),
            image_sample=sample,
            request_question=sample.get("question", ""),
            mode=mode,
            prompt_variant="standard",
            options_text=build_options_text(sample),
        )
        for sample in samples
    ]


class StructuredCoTGenerator:
    def __init__(self, api_wrapper, invalid_retry_attempts: int = 1):
        self.api = api_wrapper
        self.invalid_retry_attempts = max(0, int(invalid_retry_attempts))

    def _build_request(
        self,
        job: GenerationJob,
        image_dir: Optional[str],
    ) -> Dict[str, Any]:
        image_paths = resolve_image_paths(job.image_sample, image_dir)
        images: List[Dict[str, str]] = []
        for path in image_paths:
            base64_data = encode_image_to_base64(path)
            if base64_data:
                images.append(
                    {
                        "base64": base64_data,
                        "mime": get_mime_type(path),
                    }
                )

        if job.custom_prompt is not None:
            prompt = job.custom_prompt
        elif images:
            prompt = COT_USER_PROMPT_WITH_IMAGE.format(
                question=job.request_question,
                options_text=job.options_text,
            )
        else:
            prompt = COT_USER_PROMPT_TEXT_ONLY.format(
                question=job.request_question,
                options_text=job.options_text,
            )

        messages = {
            "system": COT_SYSTEM_PROMPT,
            "prompt": prompt,
        }
        if images:
            messages["image_base64_list"] = images
        return {"messages": messages}

    def generate_single(
        self,
        job: GenerationJob,
        image_dir: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        question = job.output_sample.get("question", "")
        if not question or not question.strip():
            return None

        request_item = self._build_request(job, image_dir)

        full_raw_cot = ""
        parsed_cot: Optional[Dict[str, Any]] = None
        is_valid = False
        validation_reason = "No generation produced."
        total_attempts = 1 + self.invalid_retry_attempts
        attempt = 0

        for attempt in range(1, total_attempts + 1):
            try:
                raw_completion = self.api.generate(request_item)
            except Exception as error:
                if attempt == 1:
                    logger.error("API call failed for: %s... | %s", question[:80], error)
                    return None
                logger.error(
                    "Retry API call failed for: %s... | attempt %s/%s | %s",
                    question[:80],
                    attempt,
                    total_attempts,
                    error,
                )
                continue

            full_raw_cot = _assemble_raw_cot(job.prefix_steps, raw_completion)
            parsed_cot = parse_cot(full_raw_cot)
            is_valid, validation_reason = validate_cot(parsed_cot)
            if not is_valid:
                debug_item = {
                    "question": question,
                    "mode": job.mode,
                    "prompt_variant": job.prompt_variant,
                    "validation_reason": validation_reason,
                    "raw_completion": raw_completion,
                    "full_raw_cot": full_raw_cot,
                }
                with open("debug_invalid_cot.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(debug_item, ensure_ascii=False) + "\n")
            if is_valid:
                break

            if attempt < total_attempts:
                logger.warning(
                    "Invalid CoT (%s), retrying... (%s/%s)",
                    validation_reason,
                    attempt + 1,
                    total_attempts,
                )
            else:
                logger.warning(
                    "Invalid CoT (%s) after %s attempts.",
                    validation_reason,
                    total_attempts,
                )

        empty_step = {"text": ""}
        empty_reasoning = {"text": "", "answer": ""}

        return {
            **job.output_sample,
            "cot": {
                "raw": full_raw_cot,
                "visual_recognition": (
                    parsed_cot.get("visual_recognition", empty_step)
                    if parsed_cot
                    else empty_step
                ),
                "knowledge_recall": (
                    parsed_cot.get("knowledge_recall", empty_step)
                    if parsed_cot
                    else empty_step
                ),
                "reasoning": (
                    parsed_cot.get("reasoning", empty_reasoning)
                    if parsed_cot
                    else empty_reasoning
                ),
            },
            "cot_meta": {
                "model": self.api.model_path,
                "mode": job.mode,
                "prompt_variant": job.prompt_variant,
                "generation_attempts": attempt,
                "invalid_retry_attempts": self.invalid_retry_attempts,
                **job.meta,
            },
            "cot_valid": is_valid,
            "cot_validation_reason": validation_reason,
        }


def process_batch(
    generator: StructuredCoTGenerator,
    jobs: List[GenerationJob],
    image_dir: Optional[str],
    output_file: str,
    num_workers: int = 4,
) -> List[Dict[str, Any]]:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_sample_ids = load_completed_sample_ids(output_file)
    previously_completed = len(completed_sample_ids)
    if previously_completed > 0:
        logger.info("Resuming: found %s already-completed samples", previously_completed)

    remaining_jobs = [
        job
        for job in jobs
        if _require_sample_identifier(job.output_sample) not in completed_sample_ids
    ]
    skipped_count = len(jobs) - len(remaining_jobs)
    if skipped_count > 0:
        logger.info(
            "Skipping %s already-completed, %s remaining",
            skipped_count,
            len(remaining_jobs),
        )
    if not remaining_jobs:
        logger.info("All samples already completed.")
        return []

    results = []
    total = len(remaining_jobs)
    success_count = 0
    fail_count = 0

    with open(output_path, "a", encoding="utf-8") as out_file:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(generator.generate_single, job, image_dir): index
                for index, job in enumerate(remaining_jobs)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                job = remaining_jobs[index]
                try:
                    result = future.result()
                    if result is None:
                        failed_record = {
                            **job.output_sample,
                            "cot_valid": False,
                            "generation_status": "api_failed",
                        }
                        out_file.write(json.dumps(failed_record, ensure_ascii=False) + "\n")
                        out_file.flush()
                        os.fsync(out_file.fileno())
                        fail_count += 1
                    elif result.get("cot_valid", False):
                        results.append(result)
                        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_file.flush()
                        os.fsync(out_file.fileno())
                        success_count += 1
                    else:
                        result["generation_status"] = "invalid_cot"
                        results.append(result)
                        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_file.flush()
                        os.fsync(out_file.fileno())
                        fail_count += 1
                except Exception as error:
                    logger.error("Error processing sample %s: %s", index, error)
                    failed_record = {
                        **job.output_sample,
                        "cot_valid": False,
                        "generation_status": "exception",
                    }
                    out_file.write(json.dumps(failed_record, ensure_ascii=False) + "\n")
                    out_file.flush()
                    os.fsync(out_file.fileno())
                    fail_count += 1

                processed = success_count + fail_count
                if processed % 10 == 0 or processed == total:
                    logger.info(
                        "Progress: %s/%s | Success: %s | Failed: %s",
                        processed,
                        total,
                        success_count,
                        fail_count,
                    )

    logger.info(
        "\n%s\nCoT Generation Complete!\n"
        "  Previously done:  %s\n"
        "  Processed now:    %s\n"
        "  Successful:       %s\n"
        "  Failed:           %s\n"
        "  Output file:      %s\n%s",
        "=" * 60,
        previously_completed,
        total,
        success_count,
        fail_count,
        output_file,
        "=" * 60,
    )

    return results


def _split_pending_jobs(
    jobs: List[GenerationJob],
    output_file: str,
) -> Tuple[set, List[GenerationJob]]:
    completed_sample_ids = load_completed_sample_ids(output_file)
    remaining_jobs = [
        job
        for job in jobs
        if _require_sample_identifier(job.output_sample) not in completed_sample_ids
    ]
    return completed_sample_ids, remaining_jobs


def _limit_jobs(
    jobs: List[GenerationJob],
    max_samples: Optional[int],
) -> List[GenerationJob]:
    if max_samples and max_samples < len(jobs):
        logger.info("Limiting to %s samples", max_samples)
        return jobs[:max_samples]
    return jobs


def _resolve_local_backend_workers(
    config: Dict[str, Any],
    requested_workers: int,
    mode: str,
) -> int:
    return requested_workers


def _run_gold_generation(
    config: Dict[str, Any],
    paths: Dict[str, Any],
    image_dir: Optional[str],
    split_filter: Optional[str],
    num_workers: int,
    max_samples: Optional[int],
):
    generation_config = get_generation_request_config(config)
    retry_attempts = int(generation_config.get("retry_attempts", 50))
    invalid_retry_attempts = max(0, retry_attempts)
    api_key = config["api"].get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("API key not provided.")
        sys.exit(1)

    samples = load_input_data(
        config["paths"]["input_file"],
        split_filter=split_filter,
    )
    jobs = _limit_jobs(build_standard_jobs(samples, mode="gold"), max_samples)

    gold_models = config["models"]["gold_cot"]
    if isinstance(gold_models, str):
        gold_models = [gold_models]

    for model_name in gold_models:
        logger.info("Generating Gold CoT candidates with model: %s", model_name)
        output_file = os.path.join(
            paths["results_dataset_dir"],
            "gold",
            model_name,
            "candidates.jsonl",
        )
        _, pending_jobs = _split_pending_jobs(jobs, output_file)
        if not pending_jobs:
            logger.info(
                "Gold CoT candidates already complete for model '%s'; "
                "skipping generator initialization.",
                model_name,
            )
            continue

        generator = StructuredCoTGenerator(
            CoTAPIWrapper(
                model_path=model_name,
                base_url=config["api"]["base_url"],
                api_key=api_key,
                max_tokens=int(generation_config.get("max_tokens", 16384)),
                timeout=float(generation_config.get("timeout", 240.0)),
                temperature=float(generation_config.get("temperature", 0.2)),
                top_p=float(generation_config.get("top_p", 0.2)),
            ),
            invalid_retry_attempts=invalid_retry_attempts,
        )
        process_batch(
            generator=generator,
            jobs=pending_jobs,
            image_dir=image_dir,
            output_file=output_file,
            num_workers=num_workers,
        )


def _run_model_generation(
    config: Dict[str, Any],
    paths: Dict[str, Any],
    image_dir: Optional[str],
    num_workers: int,
    max_samples: Optional[int],
):
    generation_config = get_generation_request_config(config)
    retry_attempts = int(generation_config.get("retry_attempts", 50))
    invalid_retry_attempts = max(0, retry_attempts)
    model_name = config["models"]["model_cot"]
    samples = load_questions_from_gold_cot(paths["gold_cot_file"])
    jobs = _limit_jobs(build_standard_jobs(samples, mode="model"), max_samples)
    _, pending_jobs = _split_pending_jobs(jobs, paths["model_cot_file"])
    if not pending_jobs:
        logger.info(
            "Model CoT generation already complete at %s; "
            "skipping model initialization.",
            paths["model_cot_file"],
        )
        return

    generator = StructuredCoTGenerator(
        create_model_cot_wrapper(
            config=config,
            model_name=model_name,
            max_tokens=int(generation_config.get("max_tokens", 16384)),
        ),
        invalid_retry_attempts=invalid_retry_attempts,
    )
    process_batch(
        generator=generator,
        jobs=pending_jobs,
        image_dir=image_dir,
        output_file=paths["model_cot_file"],
        num_workers=num_workers,
    )


def _run_replace_generation(
    config: Dict[str, Any],
    paths: Dict[str, Any],
    image_dir: Optional[str],
    num_workers: int,
    max_samples: Optional[int],
    experiments: Sequence[str],
):
    generation_config = get_generation_request_config(config)
    retry_attempts = int(generation_config.get("retry_attempts", 50))
    invalid_retry_attempts = max(0, retry_attempts)
    gold_data = load_jsonl(paths["gold_cot_file"])
    model_data = load_jsonl(paths["model_cot_file"])
    aligned_pairs = align_samples(gold_data, model_data)
    if not aligned_pairs:
        logger.error("No aligned sample pairs found.")
        sys.exit(1)

    if max_samples and max_samples < len(aligned_pairs):
        aligned_pairs = aligned_pairs[:max_samples]
        logger.info("Limiting to %s aligned samples", max_samples)

    pending_runs = []
    for experiment in experiments:
        jobs = build_replace_jobs(aligned_pairs, experiment)
        output_file = get_replace_generation_file(paths, experiment)
        _, pending_jobs = _split_pending_jobs(jobs, output_file)
        if not pending_jobs:
            logger.info(
                "Replace generation already complete for '%s'; "
                "skipping this variant.",
                experiment,
            )
            continue
        pending_runs.append((experiment, pending_jobs, output_file))

    if not pending_runs:
        logger.info(
            "All requested replace generations are already complete; "
            "skipping model initialization."
        )
        return

    generator = StructuredCoTGenerator(
        create_model_cot_wrapper(
            config=config,
            model_name=config["models"]["model_cot"],
            max_tokens=int(generation_config.get("max_tokens", 16384)),
        ),
        invalid_retry_attempts=invalid_retry_attempts,
    )

    for experiment, pending_jobs, output_file in pending_runs:
        logger.info("%s\nRunning generation mode: %s\n%s", "=" * 40, experiment, "=" * 40)
        process_batch(
            generator=generator,
            jobs=pending_jobs,
            image_dir=image_dir,
            output_file=output_file,
            num_workers=num_workers,
        )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate CoT for Medical VQA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "gold",
            "model",
            "replace_all",
            "replace_vr",
            "replace_kr",
            "replace_vr_kr",
        ],
        help="Generation mode to run.",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default=None,
        help="Dataset key from the unified config registry.",
    )
    parser.add_argument(
        "--gold-model",
        type=str,
        default=None,
        help="Gold model key/name override.",
    )
    parser.add_argument(
        "--test-model",
        type=str,
        default=None,
        help="Test model key/name override.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = load_config(
        args.config,
        dataset_key=args.dataset_key,
        gold_model_key=args.gold_model,
        test_model_key=args.test_model,
    )
    paths = resolve_paths(config)

    gen_config = config.get("generation", {})
    num_workers = gen_config.get("num_workers", 4)
    max_samples = gen_config.get("max_samples")
    image_dir = config["paths"].get("image_dir")
    split_filter = config["paths"].get("split")

    if args.mode == "gold":
        _run_gold_generation(
            config=config,
            paths=paths,
            image_dir=image_dir,
            split_filter=split_filter,
            num_workers=_resolve_local_backend_workers(config, num_workers, "gold"),
            max_samples=max_samples,
        )
        return

    if args.mode == "model":
        _run_model_generation(
            config=config,
            paths=paths,
            image_dir=image_dir,
            num_workers=_resolve_local_backend_workers(config, num_workers, "model"),
            max_samples=max_samples,
        )
        return

    if args.mode == "replace_all":
        experiments = config.get("replace_experiment", {}).get(
            "experiments",
            list(REPLACE_EXPERIMENTS),
        )
    else:
        experiments = [args.mode]

    _run_replace_generation(
        config=config,
        paths=paths,
        image_dir=image_dir,
        num_workers=_resolve_local_backend_workers(
            config,
            config.get("replace_experiment", {}).get(
                "num_workers",
                num_workers,
            ),
            "replace",
        ),
        max_samples=max_samples,
        experiments=experiments,
    )


if __name__ == "__main__":
    main()
