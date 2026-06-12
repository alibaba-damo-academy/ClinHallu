import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    extract_step_text,
    get_judge_request_config,
    get_replace_generation_file,
    get_replace_step_hallucination_judge_file,
    get_sample_identifier,
    load_completed_sample_ids,
    load_config,
    load_jsonl,
    resolve_model_api_config,
    resolve_paths,
)
from model_evaluation.step_hallucination_judge import judge_step_hallucination


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

TARGET_STEP_MAP = {
    "model": ("visual_recognition", "knowledge_recall", "reasoning"),
    "replace_vr": ("knowledge_recall", "reasoning"),
    "replace_kr": ("visual_recognition", "reasoning"),
    "replace_vr_kr": ("reasoning",),
}


def _summary_file_for_output(output_file: str) -> str:
    path = Path(output_file)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_summary.json"))
    return str(path.with_name(f"{path.name}_summary.json"))


def _require_sample_identifier(sample: Dict[str, Any]) -> str:
    sample_id = get_sample_identifier(sample)
    if sample_id is None:
        raise ValueError(
            "Sample is missing source_id/original_index and cannot be tracked by ID: "
            f"question={sample.get('question', '')!r}"
        )
    return sample_id


def _reference_by_sample_id(gold_samples: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for sample in gold_samples:
        sample_id = _require_sample_identifier(sample)
        result[sample_id] = sample
    return result


def _resolve_target_input_output(
    paths: Dict[str, Any],
    target: str,
) -> Tuple[str, str]:
    if target == "model":
        return paths["model_cot_file"], paths["step_hallucination_judge_file"]
    return (
        get_replace_generation_file(paths, target),
        get_replace_step_hallucination_judge_file(paths, target),
    )


def _build_step_result(
    judge_api: CoTAPIWrapper,
    sample: Dict[str, Any],
    reference_sample: Dict[str, Any],
    step_key: str,
    image_dir: Optional[str],
    retry_attempts: int,
) -> Dict[str, Any]:
    candidate_cot = sample.get("cot", {})
    reference_cot = reference_sample.get("gold_cot", {})
    candidate_text = extract_step_text(candidate_cot, step_key)
    reference_text = extract_step_text(reference_cot, step_key)

    return judge_step_hallucination(
        judge_api=judge_api,
        step_key=step_key,
        question=sample.get("question", ""),
        ground_truth=sample.get("answer", ""),
        candidate_text=candidate_text,
        reference_text=reference_text,
        reference_sample=reference_sample,
        candidate_sample=sample,
        image_dir=image_dir,
        max_attempts=max(1, retry_attempts + 1),
    )


def judge_single_sample(
    judge_api: CoTAPIWrapper,
    sample: Dict[str, Any],
    reference_sample: Dict[str, Any],
    target: str,
    image_dir: Optional[str],
    retry_attempts: int,
) -> Dict[str, Any]:
    sample_id = _require_sample_identifier(sample)
    step_results = {}
    for step_key in TARGET_STEP_MAP[target]:
        step_results[step_key] = _build_step_result(
            judge_api=judge_api,
            sample=sample,
            reference_sample=reference_sample,
            step_key=step_key,
            image_dir=image_dir,
            retry_attempts=retry_attempts,
        )

    return {
        "sample_id": sample_id,
        "source_id": sample.get("source_id"),
        "original_index": sample.get("original_index"),
        "question": sample.get("question", ""),
        "answer": sample.get("answer", ""),
        "target": target,
        "cot_valid": sample.get("cot_valid"),
        "generation_status": sample.get("generation_status"),
        "step_judgments": step_results,
    }


def process_step_judge_batch(
    judge_api: CoTAPIWrapper,
    samples: List[Dict[str, Any]],
    reference_by_id: Dict[str, Dict[str, Any]],
    output_file: str,
    target: str,
    image_dir: Optional[str],
    num_workers: int = 4,
    retry_attempts: int = 50,
) -> List[Dict[str, Any]]:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_sample_ids = load_completed_sample_ids(output_file)
    previously_completed = len(completed_sample_ids)
    if previously_completed > 0:
        logger.info("Resuming: found %s already-judged samples", previously_completed)

    remaining_samples = [
        sample
        for sample in samples
        if _require_sample_identifier(sample) not in completed_sample_ids
    ]
    skipped_count = len(samples) - len(remaining_samples)
    if skipped_count > 0:
        logger.info(
            "Skipping %s already-judged, %s remaining",
            skipped_count,
            len(remaining_samples),
        )
    if not remaining_samples:
        logger.info("All samples already step-judged.")
        return []

    results = []
    total = len(remaining_samples)
    processed_count = 0
    written_count = 0
    issue_count = 0

    with open(output_path, "a", encoding="utf-8") as out_file:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {}
            for index, sample in enumerate(remaining_samples):
                sample_id = _require_sample_identifier(sample)
                reference_sample = reference_by_id.get(sample_id)
                if reference_sample is None:
                    result = {
                        "sample_id": sample_id,
                        "source_id": sample.get("source_id"),
                        "original_index": sample.get("original_index"),
                        "question": sample.get("question", ""),
                        "answer": sample.get("answer", ""),
                        "target": target,
                        "cot_valid": sample.get("cot_valid"),
                        "generation_status": sample.get("generation_status"),
                        "step_judgments": {
                            step_key: {
                                "hallucinated": 1,
                                "reason": "Missing gold reference sample for this ID.",
                            }
                            for step_key in TARGET_STEP_MAP[target]
                        },
                    }
                    results.append(result)
                    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_file.flush()
                    processed_count += 1
                    issue_count += 1
                    if processed_count % 10 == 0 or processed_count == total:
                        logger.info(
                            "Progress: %s/%s | Written: %s | Immediate issues: %s",
                            processed_count,
                            total,
                            written_count,
                            issue_count,
                        )
                    continue

                future_to_index[
                    executor.submit(
                        judge_single_sample,
                        judge_api,
                        sample,
                        reference_sample,
                        target,
                        image_dir,
                        retry_attempts,
                    )
                ] = index

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                sample = remaining_samples[index]
                had_issue = False
                try:
                    result = future.result()
                except Exception as error:
                    logger.error("Error step-judging sample %s: %s", index, error)
                    had_issue = True
                    result = {
                        "sample_id": _require_sample_identifier(sample),
                        "source_id": sample.get("source_id"),
                        "original_index": sample.get("original_index"),
                        "question": sample.get("question", ""),
                        "answer": sample.get("answer", ""),
                        "target": target,
                        "cot_valid": sample.get("cot_valid"),
                        "generation_status": sample.get("generation_status"),
                        "step_judgments": {
                            step_key: {
                                "hallucinated": 1,
                                "reason": str(error),
                            }
                            for step_key in TARGET_STEP_MAP[target]
                        },
                    }

                results.append(result)
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_file.flush()
                os.fsync(out_file.fileno())

                processed_count += 1
                written_count += 1
                if had_issue:
                    issue_count += 1

                if processed_count % 10 == 0 or processed_count == total:
                    logger.info(
                        "Progress: %s/%s | Written: %s | Immediate issues: %s",
                        processed_count,
                        total,
                        written_count,
                        issue_count,
                    )

    logger.info(
        "\n%s\nStep Hallucination Judge Complete!\n"
        "  Previously done:  %s\n"
        "  Processed now:    %s\n"
        "  Written:          %s\n"
        "  Immediate issues: %s\n"
        "  Output file:      %s\n%s",
        "=" * 60,
        previously_completed,
        total,
        written_count,
        issue_count,
        output_file,
        "=" * 60,
    )
    return results


def summarize_step_judgments(records: Sequence[Dict[str, Any]], target: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "target": target,
        "total_samples": len(records),
        "steps": {},
    }
    for step_key in TARGET_STEP_MAP[target]:
        step_records = [
            record.get("step_judgments", {}).get(step_key)
            for record in records
            if step_key in record.get("step_judgments", {})
        ]
        total = len(step_records)
        hallucinated_count = sum(
            int(step_record.get("hallucinated", 1))
            for step_record in step_records
            if isinstance(step_record, dict)
        )
        summary["steps"][step_key] = {
            "total": total,
            "hallucinated": hallucinated_count,
            "hallucination_rate": round(hallucinated_count / total, 4) if total else 0.0,
        }
    return summary


def save_step_judgment_summary(output_file: str, target: str) -> Optional[Dict[str, Any]]:
    output_path = Path(output_file)
    if not output_path.exists():
        return None

    records = load_jsonl(str(output_path))
    summary = summarize_step_judgments(records, target)
    summary_path = Path(_summary_file_for_output(output_file))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, ensure_ascii=False, indent=2)
    logger.info("Step hallucination summary saved to %s", summary_path)
    return summary


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge evaluation for step-level hallucination"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["model", "replace_all", *REPLACE_EXPERIMENTS],
        help="Prediction target to evaluate.",
    )
    parser.add_argument("--dataset-key", type=str, default=None,
                        help="Dataset key from the unified config registry.")
    parser.add_argument("--gold-model", type=str, default=None,
                        help="Gold model key/name override.")
    parser.add_argument("--test-model", type=str, default=None,
                        help="Test model key/name override.")
    return parser.parse_args()


def _run_target(
    judge_api: CoTAPIWrapper,
    config: Dict[str, Any],
    paths: Dict[str, Any],
    target: str,
    reference_by_id: Dict[str, Dict[str, Any]],
) -> None:
    input_file, output_file = _resolve_target_input_output(paths, target)
    samples = load_jsonl(input_file)
    if not samples:
        logger.warning("No samples found for target '%s' at %s", target, input_file)
        return

    judge_request_config = get_judge_request_config(config)
    num_workers = int(config.get("judge", {}).get("num_workers", 4))
    process_step_judge_batch(
        judge_api=judge_api,
        samples=samples,
        reference_by_id=reference_by_id,
        output_file=output_file,
        target=target,
        image_dir=config["paths"].get("image_dir"),
        num_workers=max(1, num_workers),
        retry_attempts=int(judge_request_config.get("retry_attempts", 50)),
    )
    save_step_judgment_summary(output_file, target)


def main():
    args = parse_arguments()
    config = load_config(
        args.config,
        dataset_key=args.dataset_key,
        gold_model_key=args.gold_model,
        test_model_key=args.test_model,
    )
    paths = resolve_paths(config)

    judge_api_config = resolve_model_api_config(
        config,
        config["models"]["judge"],
        purpose="judge",
    )
    judge_request_config = get_judge_request_config(config)
    if not judge_api_config["api_key"]:
        logger.error("API key not provided.")
        sys.exit(1)

    judge_api = CoTAPIWrapper(
        model_path=judge_api_config["model_name"],
        base_url=judge_api_config["base_url"],
        api_key=judge_api_config["api_key"],
        max_tokens=int(judge_request_config.get("max_tokens", 16384)),
        timeout=float(judge_request_config.get("timeout", 240.0)),
        temperature=float(judge_request_config.get("temperature", 0.2)),
        top_p=float(judge_request_config.get("top_p", 0.2)),
    )

    gold_samples = load_jsonl(paths["gold_cot_file"])
    reference_by_id = _reference_by_sample_id(gold_samples)

    if args.target == "replace_all":
        for target in REPLACE_EXPERIMENTS:
            _run_target(judge_api, config, paths, target, reference_by_id)
    else:
        _run_target(judge_api, config, paths, args.target, reference_by_id)


if __name__ == "__main__":
    main()
