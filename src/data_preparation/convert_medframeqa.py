"""
Convert MedFrameQA dataset to the medhalu pipeline format.

MedFrameQA is a multi-image medical VQA benchmark with 2,851 samples.
Each question has 2-5 associated medical images and is multiple-choice.

MedFrameQA original format (from HuggingFace SuhaoYu1020/MedFrameQA):
    {
        "question_id": "brain_traumatic_brain_injury_CT__2oN3H0rc5Q_8",
        "system": "central_nervous_system",
        "organ": "brain",
        "keyword": "traumatic_brain_injury_CT",
        "modality": "CT",
        "video_id": "_2oN3H0rc5Q",
        "question": "Based on the CT findings...",
        "options": ["A. ...", "B. ...", "C. ...", ...],
        "correct_answer": "B",
        "image_url": ["images/xxx_1.jpg", "images/xxx_2.jpg"],
        "reasoning_chain": "The correct answer is...",
        "image_1": <PIL Image>,  (HF Image column, embedded in Parquet)
        "image_2": <PIL Image>,
        ...
    }

Converted format (compatible with medhalu pipeline):
    {
        "question": "Based on the CT findings...",
        "answer": "B. Deep left cerebral peduncle axonal injury and ...",
        "answer_letter": "B",
        "options": ["A. ...", "B. ...", "C. ...", ...],
        "images": ["brain_xxx_1.jpg", "brain_xxx_2.jpg"],
        "question_type": "multiple_choice",
        "source_id": "brain_traumatic_brain_injury_CT__2oN3H0rc5Q_8",
        "system": "central_nervous_system",
        "organ": "brain",
        "modality": "CT",
        "keyword": "traumatic_brain_injury_CT",
        ...
    }

Usage:
    # Download the dataset snapshot from HuggingFace and convert
    python scripts/convert_medframeqa.py

    # Convert from existing JSON file (already downloaded)
    python scripts/convert_medframeqa.py --input path/to/MedFrameQA.json

    # Convert from Parquet files (with embedded images)
    python scripts/convert_medframeqa.py --input path/to/parquet_dir --from-parquet

    # Increase timeout / retries for unstable networks
    python scripts/convert_medframeqa.py --download-timeout 1800 --download-attempts 8

Output:
    data/medframeqa/input/test/converted.jsonl
    data/medframeqa/input/test/images/  (extracted from Parquet or downloaded)
"""

import base64
import json
import logging
import os
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Image columns in MedFrameQA Parquet (image_1 through image_5)
IMAGE_COLUMN_NAMES = [f"image_{i}" for i in range(1, 6)]
DEFAULT_HF_DOWNLOAD_TIMEOUT = 600
DEFAULT_HF_ETAG_TIMEOUT = 120
DEFAULT_DOWNLOAD_ATTEMPTS = 5
DEFAULT_DOWNLOAD_RETRY_SLEEP_SECONDS = 15
DEFAULT_SNAPSHOT_MAX_WORKERS = 1


def split_output_paths(output_dir: str, split: str = "test") -> tuple[Path, Path, Path]:
    """Returns (<split_dir>, <converted_file>, <images_dir>) for a dataset split."""
    split_dir = Path(output_dir) / split
    converted_file = split_dir / "converted.jsonl"
    images_dir = split_dir / "images"
    return split_dir, converted_file, images_dir


def should_skip_existing(converted_file: Path, force: bool = False) -> bool:
    """Returns True if a non-empty converted file already exists and force is off."""
    return (not force) and converted_file.exists() and converted_file.stat().st_size > 0


def _set_env_temporarily(overrides: Dict[str, str]) -> Dict[str, Optional[str]]:
    previous: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    return previous


def _restore_env(previous: Dict[str, Optional[str]]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def download_medframeqa_snapshot(
    local_dir: Path,
    force: bool = False,
    *,
    download_timeout: int = DEFAULT_HF_DOWNLOAD_TIMEOUT,
    etag_timeout: int = DEFAULT_HF_ETAG_TIMEOUT,
    max_workers: int = DEFAULT_SNAPSHOT_MAX_WORKERS,
    attempts: int = DEFAULT_DOWNLOAD_ATTEMPTS,
    retry_sleep_seconds: int = DEFAULT_DOWNLOAD_RETRY_SLEEP_SECONDS,
) -> Path:
    """Download the MedFrameQA parquet snapshot with longer timeouts and retries."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: 'huggingface_hub' library not installed.")
        print("Please install it first:  pip install huggingface_hub")
        raise

    local_dir.mkdir(parents=True, exist_ok=True)
    env_backup = _set_env_temporarily(
        {
            "HF_HUB_DOWNLOAD_TIMEOUT": str(download_timeout),
            "HF_HUB_ETAG_TIMEOUT": str(etag_timeout),
        }
    )
    last_error: Optional[Exception] = None

    try:
        for attempt_index in range(1, attempts + 1):
            try:
                logger.info(
                    "Downloading MedFrameQA snapshot (attempt %s/%s) to %s",
                    attempt_index,
                    attempts,
                    local_dir,
                )
                snapshot_path = snapshot_download(
                    repo_id="SuhaoYu1020/MedFrameQA",
                    repo_type="dataset",
                    local_dir=str(local_dir),
                    allow_patterns=["*.parquet", "*.json", "README.md"],
                    max_workers=max_workers,
                    etag_timeout=etag_timeout,
                    force_download=force,
                )
                return Path(snapshot_path)
            except KeyboardInterrupt:
                raise
            except Exception as error:
                last_error = error
                logger.warning(
                    "MedFrameQA download failed (attempt %s/%s): %s",
                    attempt_index,
                    attempts,
                    error,
                )
                if attempt_index < attempts:
                    sleep_seconds = min(retry_sleep_seconds * attempt_index, 120)
                    logger.info("Sleeping %s seconds before retry...", sleep_seconds)
                    time.sleep(sleep_seconds)
    finally:
        _restore_env(env_backup)

    raise RuntimeError("Failed to download MedFrameQA snapshot") from last_error


def extract_answer_text(options: List[str], correct_answer: str) -> str:
    """Extracts the full answer text from options using the answer letter.

    Args:
        options: List of option strings, e.g. ["A. Pneumonia", "B. TB", ...]
                 or ["Pneumonia", "TB", ...] (without letter prefix).
        correct_answer: The correct answer letter, e.g. "B".

    Returns:
        Full answer string, e.g. "B. Deep left cerebral peduncle ...".
    """
    if not correct_answer or not options:
        return correct_answer

    letter = correct_answer.strip().upper()
    letter_index = ord(letter) - ord("A")

    # Try matching by letter prefix first (e.g. "B. ...")
    for option in options:
        option_stripped = option.strip()
        if option_stripped.startswith(f"{letter}.") or option_stripped.startswith(f"{letter} "):
            return option_stripped
        # Handle "(B)" prefix
        if option_stripped.startswith(f"({letter})"):
            return option_stripped

    # Fall back to index-based lookup
    if 0 <= letter_index < len(options):
        option_text = options[letter_index].strip()
        # If option doesn't have letter prefix, add it
        if not (len(option_text) >= 2 and option_text[0].isalpha() and option_text[1] in ".) "):
            return f"{letter}. {option_text}"
        return option_text

    return correct_answer


def ensure_options_have_letters(options: List[str]) -> List[str]:
    """Ensures each option has a letter prefix (A. B. C. etc.).

    If options already have letter prefixes, returns them as-is.
    Otherwise, adds letter prefixes.
    """
    if not options:
        return options

    # Check if first option already has a letter prefix
    first = options[0].strip()
    if len(first) >= 2 and first[0].isalpha() and first[1] in ".):, ":
        return options

    return [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]


def convert_sample(sample: Dict[str, Any], split: str = "test") -> Optional[Dict[str, Any]]:
    """Converts a single MedFrameQA sample to the pipeline format.

    Works with both JSON format (with image_url) and Parquet format
    (with image_1..image_5 columns, already processed).

    Args:
        sample: Raw sample dict from the dataset.
        split: Dataset split label (e.g., ``"test"``).
    """
    question_id = sample.get("question_id", "")
    question = sample.get("question", "")
    raw_options = sample.get("options", [])
    correct_answer = sample.get("correct_answer", "")

    if not question or not correct_answer:
        logger.warning(f"Skipping sample {question_id}: missing question or correct_answer")
        return None

    # Normalize options
    if isinstance(raw_options, list):
        options = ensure_options_have_letters(raw_options)
    else:
        logger.warning(f"Skipping sample {question_id}: options is not a list")
        return None

    # Build full answer text
    full_answer = extract_answer_text(options, correct_answer)

    # Build image list from image_url field (JSON format)
    # or from pre-processed images field (Parquet format)
    image_list = sample.get("images", [])
    if not image_list:
        image_urls = sample.get("image_url", [])
        if isinstance(image_urls, list):
            image_list = [url for url in image_urls if isinstance(url, str) and url]

    converted = {
        "question": question,
        "answer": full_answer,
        "answer_letter": correct_answer.strip().upper(),
        "options": options,
        "question_type": "multiple_choice",
        "source_id": question_id,
        "split": split,
    }

    if image_list:
        converted["images"] = image_list

    # Preserve image_base64_list if present (from Parquet extraction)
    if "image_base64_list" in sample:
        converted["image_base64_list"] = sample["image_base64_list"]

    # Preserve metadata fields
    for metadata_key in ("system", "organ", "keyword", "modality",
                         "video_id", "reasoning_chain"):
        if metadata_key in sample and sample[metadata_key]:
            converted[metadata_key] = sample[metadata_key]

    return converted


def convert_from_json(input_file: str, output_file: str, force: bool = False):
    """Converts a MedFrameQA JSON file to pipeline JSONL format.

    The JSON file is expected to be a list of sample dicts.
    Images are referenced by path in image_url field.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if should_skip_existing(output_path, force=force):
        logger.info(f"Skipping conversion: output already exists at {output_path}")
        return

    logger.info(f"Loading JSON from {input_path}...")

    with open(input_path, "r", encoding="utf-8") as infile:
        raw_data = json.load(infile)

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected JSON array, got {type(raw_data).__name__}")

    logger.info(f"Loaded {len(raw_data)} samples")

    converted_count = 0
    skipped_count = 0

    with open(output_path, "w", encoding="utf-8") as outfile:
        for sample in raw_data:
            converted = convert_sample(sample)
            if converted:
                outfile.write(json.dumps(converted, ensure_ascii=False) + "\n")
                converted_count += 1
            else:
                skipped_count += 1

    logger.info(f"Converted {converted_count} samples, skipped {skipped_count}")
    logger.info(f"Output saved to: {output_path}")


def convert_from_parquet(input_path: str, output_dir: str, force: bool = False):
    """Converts MedFrameQA Parquet files (with embedded images) to pipeline format.

    Extracts images from image_1..image_5 columns and saves them to disk.
    Outputs a JSONL file with image path references.

    Args:
        input_path: Path to a Parquet file or directory of Parquet files.
        output_dir: Output directory for converted.jsonl and images/.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: 'pyarrow' is required for Parquet conversion.")
        print("Install it with: pip install pyarrow")
        return

    input_dir = Path(input_path)
    split_dir, converted_file, images_dir = split_output_paths(output_dir, "test")
    split_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if should_skip_existing(converted_file, force=force):
        logger.info(f"Skipping conversion: output already exists at {converted_file}")
        return

    # Find Parquet files
    if input_dir.is_dir():
        parquet_files = sorted(input_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {input_dir}")
        logger.info(f"Found {len(parquet_files)} Parquet files in {input_dir}")
    else:
        parquet_files = [input_dir]

    converted_count = 0
    skipped_count = 0
    images_saved = 0

    with open(converted_file, "w", encoding="utf-8") as outfile:
        for parquet_path in parquet_files:
            logger.info(f"Processing {parquet_path.name}...")
            table = pq.read_table(str(parquet_path))
            columns = table.column_names
            logger.info(f"  Columns: {columns} ({len(table)} rows)")

            for row_idx in range(len(table)):
                sample = {}
                image_filenames = []

                for col_name in columns:
                    cell = table.column(col_name)[row_idx].as_py()

                    if col_name in IMAGE_COLUMN_NAMES and cell is not None:
                        # HF Image column: {"bytes": b"...", "path": "..."}
                        question_id = table.column("question_id")[row_idx].as_py()
                        image_index = col_name.split("_")[1]
                        image_filename = f"{question_id}_{image_index}.jpg"

                        if isinstance(cell, dict):
                            image_bytes = cell.get("bytes")
                            if image_bytes:
                                dest = images_dir / image_filename
                                if not dest.exists():
                                    with open(dest, "wb") as img_file:
                                        img_file.write(image_bytes)
                                    images_saved += 1
                                image_filenames.append(image_filename)
                        elif isinstance(cell, bytes):
                            dest = images_dir / image_filename
                            if not dest.exists():
                                with open(dest, "wb") as img_file:
                                    img_file.write(cell)
                                images_saved += 1
                            image_filenames.append(image_filename)
                    elif col_name not in IMAGE_COLUMN_NAMES:
                        sample[col_name] = cell

                # Set images as extracted filenames
                if image_filenames:
                    sample["images"] = image_filenames

                converted = convert_sample(sample)
                if converted:
                    # Remove image_base64_list since we saved images to disk
                    converted.pop("image_base64_list", None)
                    outfile.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    converted_count += 1
                else:
                    skipped_count += 1

                if (converted_count + skipped_count) % 200 == 0:
                    logger.info(f"  Processed {converted_count + skipped_count} samples...")

    logger.info(f"Converted {converted_count} samples, skipped {skipped_count}")
    logger.info(f"Saved {images_saved} images to {images_dir}")
    logger.info(f"Output: {converted_file}")


def download_and_convert(
    output_dir: str = "data/medframeqa/input",
    force: bool = False,
    download_dir: Optional[str] = None,
    download_timeout: int = DEFAULT_HF_DOWNLOAD_TIMEOUT,
    etag_timeout: int = DEFAULT_HF_ETAG_TIMEOUT,
    max_workers: int = DEFAULT_SNAPSHOT_MAX_WORKERS,
    attempts: int = DEFAULT_DOWNLOAD_ATTEMPTS,
    retry_sleep_seconds: int = DEFAULT_DOWNLOAD_RETRY_SLEEP_SECONDS,
):
    """Downloads MedFrameQA from HuggingFace and converts to pipeline format.

    This downloads the dataset snapshot with parquet shards, then converts
    the local parquet files to pipeline-compatible JSONL + images.
    """
    split_dir, converted_file, images_dir = split_output_paths(output_dir, "test")
    split_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if should_skip_existing(converted_file, force=force):
        print(f"Skipping [test]: output already exists at {converted_file}")
        return

    print("=" * 60)
    print("Downloading MedFrameQA from HuggingFace...")
    print(f"Output directory: {split_dir.resolve()}")
    print("=" * 60)
    snapshot_dir = Path(download_dir) if download_dir else Path(output_dir) / "source_parquet"
    snapshot_path = download_medframeqa_snapshot(
        snapshot_dir,
        force=force,
        download_timeout=download_timeout,
        etag_timeout=etag_timeout,
        max_workers=max_workers,
        attempts=attempts,
        retry_sleep_seconds=retry_sleep_seconds,
    )
    print(f"Local snapshot: {snapshot_path}")
    print("Download complete. Converting local parquet files...")

    convert_from_parquet(str(snapshot_path), output_dir, force=force)
    print(f"Output: {converted_file}")

    print()
    print("=" * 60)
    print("Next steps:")
    print("  1. Update configs/config.yaml:")
    print('     dataset: "medframeqa"')
    print('     input_file: "data/medframeqa/input/test/converted.jsonl"')
    print('     image_dir: "data/medframeqa/input/test/images"')
    print('     split: "test"')
    print()
    print("  2. Run the pipeline:")
    print(
        "     python src/model_evaluation/01_generate_cot.py "
        "--config configs/config.yaml --mode gold"
    )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MedFrameQA to medhalu pipeline format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to existing MedFrameQA JSON file, or directory of Parquet files. "
             "If not provided, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--from-parquet",
        action="store_true",
        help="Treat --input as Parquet file(s) with embedded images. "
             "Images will be extracted to disk.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/medframeqa/input",
        help="Output directory (default: data/medframeqa/input)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if output files already exist.",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Optional local directory for the downloaded HF snapshot. "
             "Defaults to {output_dir}/source_parquet.",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=DEFAULT_HF_DOWNLOAD_TIMEOUT,
        help=f"HF file download timeout in seconds (default: {DEFAULT_HF_DOWNLOAD_TIMEOUT}).",
    )
    parser.add_argument(
        "--etag-timeout",
        type=int,
        default=DEFAULT_HF_ETAG_TIMEOUT,
        help=f"HF metadata timeout in seconds (default: {DEFAULT_HF_ETAG_TIMEOUT}).",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=DEFAULT_SNAPSHOT_MAX_WORKERS,
        help=f"Concurrent snapshot download workers (default: {DEFAULT_SNAPSHOT_MAX_WORKERS}).",
    )
    parser.add_argument(
        "--download-attempts",
        type=int,
        default=DEFAULT_DOWNLOAD_ATTEMPTS,
        help=f"Snapshot download retry attempts (default: {DEFAULT_DOWNLOAD_ATTEMPTS}).",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=int,
        default=DEFAULT_DOWNLOAD_RETRY_SLEEP_SECONDS,
        help="Base sleep seconds between download retries.",
    )
    args = parser.parse_args()

    if args.input:
        if args.from_parquet:
            convert_from_parquet(args.input, args.output_dir, force=args.force)
        else:
            output_file = str(split_output_paths(args.output_dir, "test")[1])
            convert_from_json(args.input, output_file, force=args.force)
    else:
        download_and_convert(
            args.output_dir,
            force=args.force,
            download_dir=args.download_dir,
            download_timeout=args.download_timeout,
            etag_timeout=args.etag_timeout,
            max_workers=args.download_workers,
            attempts=args.download_attempts,
            retry_sleep_seconds=args.retry_sleep_seconds,
        )


if __name__ == "__main__":
    main()
