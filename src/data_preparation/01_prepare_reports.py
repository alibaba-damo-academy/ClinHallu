"""
Step 0: Image Report Generator for Medical VQA

Groups QA pairs by image, then uses an LLM to synthesize a structured
medical report for each image based on all its associated QA pairs.

This enriches the dataset with per-image context that can be used
downstream for more accurate Gold CoT filtering (Step 2).

Output:
    A JSON file mapping image identifiers to their generated reports.

Usage:
    python src/data_preparation/01_prepare_reports.py --config configs/config.yaml
"""

import json
import logging
import os
import sys
import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
for candidate in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from common import (
    CoTAPIWrapper,
    encode_image_to_base64,
    get_generation_request_config,
    get_mime_type,
    load_config,
    load_input_data,
    resolve_paths,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Silence noisy HTTP request logs from OpenAI/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ============================================================
# 1. Prompts (imported from prompts/)
# ============================================================

from prompts.report_generation import (
    REPORT_SYSTEM_PROMPT,
    REPORT_USER_PROMPT_WITH_IMAGE,
    REPORT_USER_PROMPT_TEXT_ONLY,
)

# ============================================================
# 2. Group QA by Image
# ============================================================

def _get_image_key(sample: Dict) -> str:
    """Derives a stable image identifier from a sample.

    For multi-image samples, joins all filenames with ``|`` so that samples
    sharing the exact same image set are grouped together.
    """
    images = sample.get("images")
    if isinstance(images, list) and images:
        return "|".join(str(img) for img in images)

    return "unknown_image"


def group_samples_by_image(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Groups samples by their image identifier.

    Returns:
        Dict mapping image_key -> list of samples sharing that image.
    """
    groups = defaultdict(list)
    for sample in samples:
        image_key = _get_image_key(sample)
        groups[image_key].append(sample)

    # Compute distribution statistics
    group_sizes = [len(v) for v in groups.values()]
    single_qa_count = sum(1 for s in group_sizes if s == 1)
    multi_qa_count = sum(1 for s in group_sizes if s > 1)
    max_qa = max(group_sizes) if group_sizes else 0

    logger.info(
        f"Grouped {len(samples)} samples into {len(groups)} image groups "
        f"(avg {len(samples) / max(len(groups), 1):.1f} QA per image)"
    )
    logger.info(
        f"  Distribution: {single_qa_count} images with 1 QA, "
        f"{multi_qa_count} images with 2+ QA (max {max_qa})"
    )

    # Detailed breakdown
    from collections import Counter
    size_counts = Counter(group_sizes)
    breakdown = ", ".join(
        f"{count}×{num_qa}QA" for num_qa, count in sorted(size_counts.items())
    )
    logger.info(f"  Breakdown: {breakdown}")

    return dict(groups)

# ============================================================
# 3. Report Generator
# ============================================================

def _format_qa_pairs(samples: List[Dict]) -> str:
    """Formats a list of QA samples into a readable text block."""
    lines = []
    for idx, sample in enumerate(samples, 1):
        question = sample.get("question", "").strip()
        answer = sample.get("answer", "").strip()
        lines.append(f"Q{idx}: {question}")
        lines.append(f"A{idx}: {answer}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_report_request(
    qa_text: str,
    image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Builds an API request for report generation.

    Args:
        qa_text: Formatted QA pairs text.
        image_paths: Optional list of local image file paths.
    """
    images: List[Dict[str, str]] = []
    if image_paths:
        for path in image_paths:
            base64_data = encode_image_to_base64(path)
            if base64_data:
                images.append({
                    "base64": base64_data,
                    "mime": get_mime_type(path),
                })

    if images:
        return {
            "messages": {
                "system": REPORT_SYSTEM_PROMPT,
                "prompt": REPORT_USER_PROMPT_WITH_IMAGE.format(qa_text=qa_text),
                "image_base64_list": images,
            }
        }

    return {
        "messages": {
            "system": REPORT_SYSTEM_PROMPT,
            "prompt": REPORT_USER_PROMPT_TEXT_ONLY.format(qa_text=qa_text),
        }
    }


def _resolve_image_paths_for_group(
    samples: List[Dict],
    image_dir: Optional[str],
) -> List[str]:
    """Resolves image file paths for a group of samples sharing the same image."""
    if not image_dir:
        return []

    # Collect unique image filenames from the group (all samples share the same images)
    filenames: List[str] = []
    for sample in samples:
        images_field = sample.get("images")
        if isinstance(images_field, list) and images_field:
            for img in images_field:
                if isinstance(img, str) and img and img not in filenames:
                    filenames.append(img)
            break  # All samples in the group share the same image(s)

    resolved = []
    for filename in filenames:
        full_path = os.path.join(image_dir, filename)
        if os.path.exists(full_path):
            resolved.append(full_path)
    return resolved


def _is_retryable_error(error: Exception) -> bool:
    """Best-effort classification for transient API failures."""
    message = str(error).lower()
    retryable_signals = (
        "timeout",
        "timed out",
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "connection",
        "temporarily unavailable",
        "server error",
    )
    return any(signal in message for signal in retryable_signals)


def generate_report_for_image(
    api: CoTAPIWrapper,
    image_key: str,
    samples: List[Dict],
    image_dir: Optional[str] = None,
    max_attempts: int = 3,
    retry_sleep_seconds: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """
    Generates a medical report for a single image group.

    Args:
        api: API wrapper instance.
        image_key: Image identifier string.
        samples: List of QA samples sharing this image.
        image_dir: Directory containing image files.

    Returns:
        Dict with image_key, report text, and metadata, or None on failure.
    """
    qa_text = _format_qa_pairs(samples)
    image_paths = _resolve_image_paths_for_group(samples, image_dir)
    request_item = _build_report_request(qa_text, image_paths)

    raw_report = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw_report = api.generate(request_item)
            break
        except Exception as error:
            is_retryable = _is_retryable_error(error)
            is_last_attempt = attempt == max_attempts
            if not is_retryable or is_last_attempt:
                logger.error(
                    "API call failed for image '%s' on attempt %s/%s: %s",
                    image_key,
                    attempt,
                    max_attempts,
                    error,
                )
                return None

            sleep_seconds = retry_sleep_seconds * attempt
            logger.warning(
                "Transient API failure for image '%s' on attempt %s/%s: %s. "
                "Retrying in %.1fs.",
                image_key,
                attempt,
                max_attempts,
                error,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    return {
        "image_key": image_key,
        "report": raw_report,
        "num_qa_pairs": len(samples),
        "qa_pairs": [
            {"question": s.get("question", ""), "answer": s.get("answer", "")}
            for s in samples
        ],
    }

# ============================================================
# 4. Batch Processing
# ============================================================

def process_reports(
    api: CoTAPIWrapper,
    image_groups: Dict[str, List[Dict]],
    output_file: str,
    image_dir: Optional[str] = None,
    num_workers: int = 4,
    max_attempts: int = 3,
    retry_sleep_seconds: float = 5.0,
) -> Dict[str, Any]:
    """Generates reports for all image groups with multi-threading.

    Args:
        api: API wrapper instance.
        image_groups: Dict mapping image_key -> list of samples.
        output_file: Path to output JSON file.
        image_dir: Directory containing image files.
        num_workers: Number of concurrent threads.

    Returns:
        Dict mapping image_key -> report data.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing reports for resume support
    existing_reports = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as file_handle:
                existing_reports = json.load(file_handle)
            logger.info(f"Resuming: found {len(existing_reports)} existing reports")
        except (json.JSONDecodeError, Exception):
            pass

    remaining_groups = {
        key: samples for key, samples in image_groups.items()
        if key not in existing_reports
    }

    if not remaining_groups:
        logger.info("All image reports already generated.")
        return existing_reports

    logger.info(
        f"Generating reports for {len(remaining_groups)} image groups "
        f"({len(existing_reports)} already done)"
    )

    reports = dict(existing_reports)
    success_count = 0
    fail_count = 0
    total = len(remaining_groups)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_key = {
            executor.submit(
                generate_report_for_image,
                api,
                image_key,
                samples,
                image_dir,
                max_attempts,
                retry_sleep_seconds,
            ): image_key
            for image_key, samples in remaining_groups.items()
        }

        for future in as_completed(future_to_key):
            image_key = future_to_key[future]
            try:
                result = future.result()
                if result:
                    reports[image_key] = result
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as error:
                logger.error(f"Error generating report for '{image_key}': {error}")
                fail_count += 1

            processed = success_count + fail_count
            if processed % 10 == 0 or processed == total:
                logger.info(
                    f"Progress: {processed}/{total} | "
                    f"Success: {success_count} | Failed: {fail_count}"
                )

            # Incremental save every 50 reports
            if processed % 50 == 0:
                with open(output_path, "w", encoding="utf-8") as file_handle:
                    json.dump(reports, file_handle, ensure_ascii=False, indent=2)

    # Final save
    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(reports, file_handle, ensure_ascii=False, indent=2)

    logger.info(
        f"\n{'=' * 60}\n"
        f"Image Report Generation Complete!\n"
        f"  Previously done:  {len(existing_reports)}\n"
        f"  Processed now:    {total}\n"
        f"  Successful:       {success_count}\n"
        f"  Failed:           {fail_count}\n"
        f"  Total reports:    {len(reports)}\n"
        f"  Output file:      {output_file}\n"
        f"{'=' * 60}"
    )

    return reports

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate Image Reports for Medical VQA"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML file.")
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

    api_key = config["api"].get("api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("API key not provided. Set api.api_key in config or OPENAI_API_KEY env var.")
        sys.exit(1)

    report_config = config.get("report", {})
    generation_request_config = get_generation_request_config(config)
    model_name = report_config.get("model", config["models"]["gold_cot"][0])
    num_workers = report_config.get("num_workers",
                                    config.get("generation", {}).get("num_workers", 4))
    timeout = float(generation_request_config.get("timeout", 240.0))
    max_attempts = int(generation_request_config.get("retry_attempts", 50))
    retry_sleep_seconds = float(report_config.get("retry_sleep_seconds", 5.0))
    output_file = paths["image_reports_file"]

    # Load input data
    samples = load_input_data(
        config["paths"]["input_file"],
        split_filter=config["paths"].get("split"),
    )

    if not samples:
        logger.error("No samples loaded.")
        sys.exit(1)

    # Group by image
    image_groups = group_samples_by_image(samples)

    # Initialize API
    api = CoTAPIWrapper(
        model_path=model_name,
        base_url=config["api"]["base_url"],
        api_key=api_key,
        max_tokens=int(generation_request_config.get("max_tokens", 16384)),
        timeout=timeout,
        temperature=float(generation_request_config.get("temperature", 0.2)),
        top_p=float(generation_request_config.get("top_p", 0.2)),
    )

    # Generate reports
    image_dir = config["paths"].get("image_dir")
    process_reports(
        api=api,
        image_groups=image_groups,
        output_file=output_file,
        image_dir=image_dir,
        num_workers=num_workers,
        max_attempts=max_attempts,
        retry_sleep_seconds=retry_sleep_seconds,
    )


if __name__ == "__main__":
    main()
