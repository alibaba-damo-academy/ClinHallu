import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
for candidate in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from model_evaluation.answer_judge import judge_answer_correctness
from common import (
    CoTAPIWrapper,
    check_answer_match,
    get_sample_identifier,
    get_judge_request_config,
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
    cot = sample.get("cot", {})
    reasoning = cot.get("reasoning", {})
    if isinstance(reasoning, dict):
        return reasoning.get("answer", "")
    return ""


def _get_image_key(sample: Dict[str, Any]) -> str:
    images = sample.get("images")
    if isinstance(images, list) and images:
        return "|".join(str(image_name) for image_name in images)
    return ""


def _try_multiple_choice_match(
    sample: Dict[str, Any],
    predicted: str,
) -> Optional[Dict[str, Any]]:
    answer_letter = sample.get("answer_letter", "")
    if not answer_letter:
        return None

    predicted_clean = predicted.strip()
    if not predicted_clean:
        return None

    predicted_letter = ""
    if predicted_clean[0].isalpha() and (
        len(predicted_clean) == 1 or not predicted_clean[1].isalpha()
    ):
        predicted_letter = predicted_clean[0].upper()

    if predicted_letter == answer_letter.upper():
        return {
            "is_correct": True,
            "verdict": "CORRECT",
            "reason": (
                f"Multiple-choice letter match: predicted '{predicted_letter}' "
                f"== ground truth '{answer_letter}'"
            ),
        }

    return None


def load_reports(reports_file: str) -> Optional[Dict[str, Any]]:
    reports_path = Path(reports_file)
    if not reports_path.exists():
        logger.info("No reports file found at %s, proceeding without reports", reports_file)
        return None

    try:
        with open(reports_path, "r", encoding="utf-8") as file_handle:
            reports = json.load(file_handle)
    except (json.JSONDecodeError, Exception) as error:
        logger.warning("Failed to load reports: %s", error)
        return None

    logger.info("Loaded %s image reports from %s", len(reports), reports_file)
    return reports


def judge_single_answer(
    api: CoTAPIWrapper,
    sample: Dict[str, Any],
    reports: Optional[Dict[str, Any]] = None,
    retry_attempts: int = 50,
) -> Dict[str, Any]:
    question = sample.get("question", "")
    ground_truth = sample.get("answer", "")
    predicted = extract_predicted_answer(sample)

    mc_result = _try_multiple_choice_match(sample, predicted)
    if mc_result is not None:
        return {
            **mc_result,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
        }

    report_text = None
    if reports:
        image_key = _get_image_key(sample)
        report_entry = reports.get(image_key)
        if isinstance(report_entry, dict):
            report_text = report_entry.get("report", "")

    result = judge_answer_correctness(
        judge_api=api,
        question=question,
        predicted_answer=predicted,
        ground_truth=ground_truth,
        report_text=report_text,
        max_attempts=max(1, retry_attempts + 1),
        fallback_to_local=True,
    )

    if result["answer_correct"] == -1:
        local_match = check_answer_match(predicted, ground_truth, 0.8)
        result = {
            "answer_correct": 1 if local_match else 0,
            "answer_reasoning": "Fallback local string matching after judge failure",
        }

    return {
        "is_correct": result["answer_correct"] == 1,
        "verdict": "CORRECT" if result["answer_correct"] == 1 else "INCORRECT",
        "reason": result.get("answer_reasoning", ""),
        "question": question,
        "ground_truth": ground_truth,
        "predicted": predicted,
    }


def _load_incremental_results(output_file: str) -> List[Dict[str, Any]]:
    return load_jsonl(output_file)


def filter_gold_cot(
    candidates: List[Dict[str, Any]],
    api: CoTAPIWrapper,
    reports: Optional[Dict[str, Any]] = None,
    num_workers: int = 4,
    retry_attempts: int = 50,
    accepted_output_file: Optional[str] = None,
    rejected_output_file: Optional[str] = None,
) -> Dict[str, Any]:
    structurally_valid = []
    rejected_invalid_cot = []

    for sample in candidates:
        is_valid = sample.get("cot_valid", False)
        ground_truth = sample.get("answer", "")
        if not is_valid or not ground_truth:
            rejected_invalid_cot.append(sample)
            continue
        structurally_valid.append(sample)

    logger.info(
        "Pass 1 (structure): %s valid, %s rejected",
        len(structurally_valid),
        len(rejected_invalid_cot),
    )

    already_judged = set()
    if accepted_output_file:
        already_judged |= load_completed_sample_ids(accepted_output_file)
    if rejected_output_file:
        already_judged |= load_completed_sample_ids(rejected_output_file)

    remaining_samples = [
        sample
        for sample in structurally_valid
        if _require_sample_identifier(sample) not in already_judged
    ]
    previously_completed = len(structurally_valid) - len(remaining_samples)
    if previously_completed > 0:
        logger.info("Resuming: found %s already-judged samples", previously_completed)
        logger.info(
            "Skipping %s already-judged, %s remaining",
            previously_completed,
            len(remaining_samples),
        )

    accepted = []
    rejected_wrong_answer = []
    total = len(remaining_samples)
    success_count = 0
    fail_count = 0

    if accepted_output_file:
        Path(accepted_output_file).parent.mkdir(parents=True, exist_ok=True)
    if rejected_output_file:
        Path(rejected_output_file).parent.mkdir(parents=True, exist_ok=True)

    accepted_fh = (
        open(accepted_output_file, "a", encoding="utf-8")
        if accepted_output_file
        else None
    )
    rejected_fh = (
        open(rejected_output_file, "a", encoding="utf-8")
        if rejected_output_file
        else None
    )

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_sample = {
                executor.submit(
                    judge_single_answer,
                    api,
                    sample,
                    reports,
                    retry_attempts,
                ): sample
                for sample in remaining_samples
            }

            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    judge_result = future.result()
                    is_correct = judge_result.get("is_correct", False)
                    predicted_answer = judge_result.get("predicted", "")

                    if is_correct:
                        gold_sample = {
                            key: value
                            for key, value in sample.items()
                            if key
                            not in (
                                "cot",
                                "cot_meta",
                                "cot_valid",
                                "cot_validation_reason",
                                "generation_status",
                            )
                        }
                        gold_sample["gold_cot"] = sample["cot"]
                        gold_sample["gold_cot_meta"] = {
                            **sample.get("cot_meta", {}),
                            "method": "natural",
                            "predicted_answer": predicted_answer,
                            "is_correct": True,
                            "judge_verdict": judge_result.get("verdict", ""),
                            "judge_reason": judge_result.get("reason", ""),
                        }
                        gold_sample["cot_valid"] = True
                        accepted.append(gold_sample)
                        if accepted_fh:
                            accepted_fh.write(json.dumps(gold_sample, ensure_ascii=False) + "\n")
                            accepted_fh.flush()
                        success_count += 1
                    else:
                        sample["judge_verdict"] = judge_result.get("verdict", "")
                        sample["judge_reason"] = judge_result.get("reason", "")
                        rejected_wrong_answer.append(sample)
                        if rejected_fh:
                            rejected_fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                            rejected_fh.flush()
                        fail_count += 1
                except Exception as error:
                    logger.error("Error judging sample: %s", error)
                    rejected_wrong_answer.append(sample)
                    fail_count += 1

                processed = success_count + fail_count
                if processed % 10 == 0 or processed == total:
                    logger.info(
                        "Pass 2 (judge): %s/%s | Accepted: %s | Rejected: %s",
                        processed,
                        total,
                        success_count,
                        fail_count,
                    )
    finally:
        if accepted_fh:
            accepted_fh.close()
        if rejected_fh:
            rejected_fh.close()

    all_accepted = (
        _load_incremental_results(accepted_output_file)
        if accepted_output_file
        else accepted
    )
    all_rejected = (
        _load_incremental_results(rejected_output_file)
        if rejected_output_file
        else rejected_wrong_answer
    )

    total_candidates = len(candidates)
    stats = {
        "total_candidates": total_candidates,
        "structurally_valid": len(structurally_valid),
        "accepted": len(all_accepted),
        "rejected_wrong_answer": len(all_rejected),
        "rejected_invalid_cot": len(rejected_invalid_cot),
        "acceptance_rate": round(len(all_accepted) / max(total_candidates, 1), 4),
        "judge_model": api.model_path,
    }

    return {
        "accepted": all_accepted,
        "rejected_wrong_answer": all_rejected,
        "rejected_invalid_cot": rejected_invalid_cot,
        "stats": stats,
    }


def save_jsonl(records: List[Dict[str, Any]], output_file: str):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Filter Gold CoT by answer correctness using LLM Judge"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file.",
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

    gold_filter_config = config.get("gold_filter", config.get("gold_cot_filter", {}))
    judge_model = gold_filter_config.get("judge_model") or config["models"]["judge"]
    num_workers = gold_filter_config.get("num_workers", 4)
    judge_request_config = get_judge_request_config(config)

    candidates = load_jsonl(paths["gold_cot_candidates_file"])
    reports = load_reports(paths["image_reports_file"])
    judge_api_config = resolve_model_api_config(
        config,
        judge_model,
        purpose="judge",
    )
    if not judge_api_config["api_key"]:
        logger.error("API key not provided.")
        sys.exit(1)
    api = CoTAPIWrapper(
        model_path=judge_api_config["model_name"],
        base_url=judge_api_config["base_url"],
        api_key=judge_api_config["api_key"],
        max_tokens=int(judge_request_config.get("max_tokens", 16384)),
        timeout=float(judge_request_config.get("timeout", 240.0)),
        temperature=float(judge_request_config.get("temperature", 0.2)),
        top_p=float(judge_request_config.get("top_p", 0.2)),
    )

    result = filter_gold_cot(
        candidates=candidates,
        api=api,
        reports=reports,
        num_workers=num_workers,
        retry_attempts=int(judge_request_config.get("retry_attempts", 50)),
        accepted_output_file=paths["gold_cot_file"],
        rejected_output_file=paths["rejected_wrong_answer_file"],
    )

    save_jsonl(result["rejected_invalid_cot"], paths["rejected_invalid_cot_file"])
    with open(paths["filter_stats_file"], "w", encoding="utf-8") as file_handle:
        json.dump(result["stats"], file_handle, ensure_ascii=False, indent=2)

    logger.info("Saved %s Gold CoT samples to %s", len(result["accepted"]), paths["gold_cot_file"])
    logger.info(
        "Saved %s rejected wrong-answer samples to %s",
        len(result["rejected_wrong_answer"]),
        paths["rejected_wrong_answer_file"],
    )
    logger.info(
        "Saved %s invalid CoT samples to %s",
        len(result["rejected_invalid_cot"]),
        paths["rejected_invalid_cot_file"],
    )
    logger.info("Saved filter stats to %s", paths["filter_stats_file"])


if __name__ == "__main__":
    main()
