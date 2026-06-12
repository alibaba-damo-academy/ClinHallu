import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
for candidate in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from model_evaluation.answer_judge import (
    DEFAULT_JUDGE_RESULT,
    judge_answer_correctness,
)
from common import (
    CoTAPIWrapper,
    REPLACE_DESCRIPTIONS,
    REPLACE_EXPERIMENTS,
    extract_answer,
    get_sample_identifier,
    get_judge_request_config,
    get_replace_generation_file,
    get_replace_judge_file,
    load_completed_sample_ids,
    load_config,
    load_jsonl,
    resolve_model_api_config,
    resolve_paths,
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


def extract_predicted_answer(sample: Dict[str, Any]) -> str:
    cot = sample.get("cot")
    if isinstance(cot, dict):
        answer = extract_answer(cot)
        if answer:
            return answer
    return sample.get("predicted_answer", "")


def judge_single_sample(
    api_wrapper: CoTAPIWrapper,
    sample: Dict[str, Any],
    target: str,
    retry_attempts: int = 50,
) -> Dict[str, Any]:
    question = sample.get("question", "")
    ground_truth = sample.get("answer") or sample.get("ground_truth", "")
    predicted_answer = extract_predicted_answer(sample)

    if not ground_truth:
        return {
            "question": question,
            "target": target,
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth,
            "answer_correct": -1,
            "answer_reasoning": "Missing ground truth answer",
            "cot_valid": sample.get("cot_valid"),
            "generation_status": sample.get("generation_status"),
            "source_id": sample.get("source_id"),
            "original_index": sample.get("original_index"),
        }

    judge_result = judge_answer_correctness(
        judge_api=api_wrapper,
        question=question,
        predicted_answer=predicted_answer,
        ground_truth=ground_truth,
        max_attempts=max(1, retry_attempts + 1),
        fallback_to_local=False,
    )

    return {
        "question": question,
        "target": target,
        "predicted_answer": predicted_answer,
        "ground_truth": ground_truth,
        "answer_correct": judge_result.get("answer_correct", -1),
        "answer_reasoning": judge_result.get("answer_reasoning", ""),
        "cot_valid": sample.get("cot_valid"),
        "generation_status": sample.get("generation_status"),
        "source_id": sample.get("source_id"),
        "original_index": sample.get("original_index"),
    }


def aggregate_judge_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"error": "No results to aggregate"}

    valid_results = [
        result for result in results if result.get("answer_correct", -1) != -1
    ]
    valid_count = len(valid_results)
    if valid_count == 0:
        return {"error": "All judge calls failed", "total_samples": total}

    cot_valid_results = [
        result for result in valid_results if result.get("cot_valid") is True
    ]
    cot_valid_count = len(cot_valid_results)
    cot_invalid_or_missing_count = valid_count - cot_valid_count
    if cot_valid_count == 0:
        return {
            "error": "No cot_valid=true judge results",
            "total_samples": total,
            "valid_judge_results": valid_count,
            "failed_judge_results": total - valid_count,
            "cot_valid_judge_results": 0,
            "cot_invalid_or_missing_judge_results": cot_invalid_or_missing_count,
        }

    answer_correct_count = sum(
        1 for result in cot_valid_results if result.get("answer_correct", 0) == 1
    )

    def safe_rate(count: int, denominator: int) -> float:
        return round(count / denominator, 4) if denominator > 0 else 0.0

    return {
        "total_samples": total,
        "valid_judge_results": valid_count,
        "failed_judge_results": total - valid_count,
        "cot_valid_judge_results": cot_valid_count,
        "cot_invalid_or_missing_judge_results": cot_invalid_or_missing_count,
        "answer_accuracy": {
            "scope": "cot_valid_true_only",
            "acc": safe_rate(answer_correct_count, cot_valid_count),
            "correct_count": answer_correct_count,
            "total": cot_valid_count,
        },
    }


def _load_all_results(output_file: str) -> List[Dict[str, Any]]:
    return load_jsonl(output_file)


def process_judge_batch(
    api_wrapper: CoTAPIWrapper,
    samples: List[Dict[str, Any]],
    output_file: str,
    target: str,
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
        logger.info("All samples already judged.")
        return []

    results = []
    total = len(remaining_samples)
    success_count = 0
    fail_count = 0

    with open(output_path, "a", encoding="utf-8") as out_file:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(
                    judge_single_sample,
                    api_wrapper,
                    sample,
                    target,
                    retry_attempts,
                ): index
                for index, sample in enumerate(remaining_samples)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                except Exception as error:
                    logger.error("Error judging sample %s: %s", index, error)
                    result = {
                        "question": remaining_samples[index].get("question", ""),
                        "target": target,
                        "predicted_answer": extract_predicted_answer(remaining_samples[index]),
                        "ground_truth": remaining_samples[index].get("answer", ""),
                        "answer_correct": DEFAULT_JUDGE_RESULT["answer_correct"],
                        "answer_reasoning": str(error),
                        "source_id": remaining_samples[index].get("source_id"),
                        "original_index": remaining_samples[index].get("original_index"),
                    }

                results.append(result)
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_file.flush()

                if result.get("answer_correct", -1) != -1:
                    success_count += 1
                else:
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
        "\n%s\nLLM Judge Complete!\n"
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


def print_judge_summary(label: str, aggregated: Dict[str, Any]):
    accuracy = aggregated.get("answer_accuracy", {})
    logger.info(
        "\n%s\n%s (%s judge-valid, %s cot-valid samples)\n%s\n"
        "ACC [cot_valid=true]: %0.4f (%s/%s)\n%s",
        "=" * 60,
        label,
        aggregated.get("valid_judge_results", 0),
        aggregated.get("cot_valid_judge_results", 0),
        "=" * 60,
        accuracy.get("acc", 0),
        accuracy.get("correct_count", 0),
        accuracy.get("total", 0),
        "=" * 60,
    )


def aggregated_to_baseline(aggregated: Dict[str, Any]) -> Dict[str, Any]:
    accuracy = aggregated.get("answer_accuracy", {})
    return {
        "scope": accuracy.get("scope", "cot_valid_true_only"),
        "correct": accuracy.get("correct_count", 0),
        "total": accuracy.get("total", 0),
        "acc": accuracy.get("acc", 0.0),
    }


def _load_json_summary(summary_file: str) -> Optional[Dict[str, Any]]:
    path = Path(summary_file)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except (json.JSONDecodeError, Exception) as error:
        logger.warning("Failed to load JSON summary %s: %s", summary_file, error)
        return None


def _write_baseline_summary(
    paths: Dict[str, Any],
    judge_model: str,
    aggregated: Dict[str, Any],
    input_file: str,
):
    replace_summary = _load_json_summary(paths["replace_summary_file"])
    output = {
        "config": {
            "dataset": paths["dataset"],
            "gold_model": paths["gold_model"],
            "model_name": paths["model_name"],
            "input_file": input_file,
            "judge_model": judge_model,
            "details_file": paths["judge_details_file"],
        },
        "aggregated_metrics": aggregated,
        "replace_experiment": replace_summary,
    }

    result_path = Path(paths["judge_result_file"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as file_handle:
        json.dump(output, file_handle, ensure_ascii=False, indent=2)
    logger.info("Aggregated results saved to %s", paths["judge_result_file"])


def _merge_replace_summary_into_baseline(
    paths: Dict[str, Any],
    replace_summary: Dict[str, Any],
):
    baseline_summary = _load_json_summary(paths["judge_result_file"])
    if baseline_summary is None:
        return

    baseline_summary["replace_experiment"] = replace_summary
    with open(paths["judge_result_file"], "w", encoding="utf-8") as file_handle:
        json.dump(baseline_summary, file_handle, ensure_ascii=False, indent=2)
    logger.info("Merged replace summary into %s", paths["judge_result_file"])


def _get_target_files(paths: Dict[str, Any], target: str) -> Dict[str, str]:
    if target == "model":
        return {
            "input_file": paths["model_cot_file"],
            "details_file": paths["judge_details_file"],
        }
    return {
        "input_file": get_replace_generation_file(paths, target),
        "details_file": get_replace_judge_file(paths, target),
    }


def _build_skipped_run(target: str, input_file: str, details_file: str) -> Dict[str, Any]:
    return {
        "target": target,
        "input_file": input_file,
        "details_file": details_file,
        "aggregated": {},
        "skipped": True,
    }


def _run_single_target(
    config: Dict[str, Any],
    paths: Dict[str, Any],
    judge_api: CoTAPIWrapper,
    target: str,
    num_workers: int,
    retry_attempts: int,
) -> Dict[str, Any]:
    target_files = _get_target_files(paths, target)
    input_file = target_files["input_file"]
    if not Path(input_file).exists():
        logger.warning(
            "Skipping target '%s' because input file does not exist: %s",
            target,
            input_file,
        )
        return _build_skipped_run(target, input_file, target_files["details_file"])

    samples = load_jsonl(input_file)
    process_judge_batch(
        api_wrapper=judge_api,
        samples=samples,
        output_file=target_files["details_file"],
        target=target,
        num_workers=num_workers,
        retry_attempts=retry_attempts,
    )
    all_results = _load_all_results(target_files["details_file"])
    aggregated = aggregate_judge_results(all_results)
    print_judge_summary(f"Judge Summary [{target}]", aggregated)
    return {
        "target": target,
        "input_file": input_file,
        "details_file": target_files["details_file"],
        "aggregated": aggregated,
        "skipped": False,
    }


def _load_or_compute_baseline(
    config: Dict[str, Any],
    paths: Dict[str, Any],
    judge_api: CoTAPIWrapper,
    num_workers: int,
    retry_attempts: int,
) -> Optional[Dict[str, Any]]:
    baseline_summary = _load_json_summary(paths["judge_result_file"])
    if baseline_summary:
        aggregated = baseline_summary.get("aggregated_metrics", {})
        if aggregated.get("answer_accuracy"):
            return aggregated_to_baseline(aggregated)

    baseline_run = _run_single_target(
        config=config,
        paths=paths,
        judge_api=judge_api,
        target="model",
        num_workers=num_workers,
        retry_attempts=retry_attempts,
    )
    if baseline_run.get("skipped"):
        logger.warning("Baseline model judge skipped because model_cot.jsonl is missing.")
        return None

    _write_baseline_summary(
        paths=paths,
        judge_model=config["models"]["judge"],
        aggregated=baseline_run["aggregated"],
        input_file=baseline_run["input_file"],
    )
    return aggregated_to_baseline(baseline_run["aggregated"])


def _save_replace_summary(
    paths: Dict[str, Any],
    baseline_acc: Optional[Dict[str, Any]],
    experiment_metrics: Dict[str, Dict[str, Any]],
):
    existing = _load_json_summary(paths["replace_summary_file"]) or {
        "baseline": baseline_acc,
        "experiments": {},
    }
    existing["baseline"] = baseline_acc

    for experiment_name, metrics in experiment_metrics.items():
        delta = (
            round(metrics["acc"] - baseline_acc["acc"], 4)
            if baseline_acc is not None
            else None
        )
        existing["experiments"][experiment_name] = {
            "scope": metrics.get("scope", "cot_valid_true_only"),
            "correct": metrics["correct"],
            "total": metrics["total"],
            "acc": metrics["acc"],
            "delta_vs_baseline": delta,
            "description": REPLACE_DESCRIPTIONS.get(experiment_name, ""),
        }

    summary_path = Path(paths["replace_summary_file"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(existing, file_handle, ensure_ascii=False, indent=2)
    logger.info("Replace summary saved to %s", paths["replace_summary_file"])
    _merge_replace_summary_into_baseline(paths, existing)


def _experiments_from_config(config: Dict[str, Any]) -> Sequence[str]:
    return config.get("replace_experiment", {}).get(
        "experiments",
        list(REPLACE_EXPERIMENTS),
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge evaluation for final answers"
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
    num_workers = config.get("judge", {}).get("num_workers", 4)
    retry_attempts = int(judge_request_config.get("retry_attempts", 50))

    if args.target == "model":
        baseline_run = _run_single_target(
            config=config,
            paths=paths,
            judge_api=judge_api,
            target="model",
            num_workers=num_workers,
            retry_attempts=retry_attempts,
        )
        if baseline_run.get("skipped"):
            return
        _write_baseline_summary(
            paths=paths,
            judge_model=config["models"]["judge"],
            aggregated=baseline_run["aggregated"],
            input_file=baseline_run["input_file"],
        )
        return

    baseline_acc = _load_or_compute_baseline(
        config=config,
        paths=paths,
        judge_api=judge_api,
        num_workers=num_workers,
        retry_attempts=retry_attempts,
    )

    if args.target == "replace_all":
        experiments = _experiments_from_config(config)
    else:
        experiments = [args.target]

    experiment_metrics = {}
    for experiment in experiments:
        run = _run_single_target(
            config=config,
            paths=paths,
            judge_api=judge_api,
            target=experiment,
            num_workers=num_workers,
            retry_attempts=retry_attempts,
        )
        if run.get("skipped"):
            continue
        aggregated = run["aggregated"]
        baseline = aggregated_to_baseline(aggregated)
        experiment_metrics[experiment] = baseline

    if not experiment_metrics:
        logger.warning("No replace judge inputs were found. Nothing to save.")
        return

    _save_replace_summary(
        paths=paths,
        baseline_acc=baseline_acc,
        experiment_metrics=experiment_metrics,
    )


if __name__ == "__main__":
    main()
