"""
Convert VQA-RAD dataset to the medhalu pipeline format.

VQA-RAD is a radiology visual question answering dataset with ~2.2k QA pairs
on ~314 radiology images. Questions are either open-ended or binary yes/no.

VQA-RAD original format (from HuggingFace flaviagiammarino/vqa-rad):
    {
        "image": <PIL Image>,
        "question": "are regions of the brain infarcted?",
        "answer": "yes"
    }

Converted format (compatible with medhalu pipeline):
    {
        "question": "are regions of the brain infarcted?",
        "answer": "yes",
        "images": ["vqarad_00001.jpg"],
        "question_type": "open_ended" | "yes_no",
        "source": "vqa-rad",
        "split": "test",
        "original_index": 0
    }

Usage:
    # Download from HuggingFace and convert (default: test split)
    python src/data_preparation/convert_vqarad.py

    # Convert from local Parquet files
    python src/data_preparation/convert_vqarad.py --from-parquet data/vqa-rad/input/data

    # Convert all splits
    python src/data_preparation/convert_vqarad.py --split all

Output:
    data/vqa-rad/input/<split>/converted.jsonl
    data/vqa-rad/input/<split>/images/
"""

import json
import logging
import os
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

YES_NO_ANSWERS = {"yes", "no"}


def split_output_paths(output_dir: str, split: str) -> tuple[Path, Path, Path]:
    """Returns (<split_dir>, <converted_file>, <images_dir>) for a dataset split."""
    split_dir = Path(output_dir) / split
    converted_file = split_dir / "converted.jsonl"
    images_dir = split_dir / "images"
    return split_dir, converted_file, images_dir


def should_skip_existing(converted_file: Path, force: bool = False) -> bool:
    """Returns True if a non-empty converted file already exists and force is off."""
    return (not force) and converted_file.exists() and converted_file.stat().st_size > 0


def classify_question_type(answer: str) -> str:
    """Classifies the question type based on the answer."""
    if answer.strip().lower() in YES_NO_ANSWERS:
        return "yes_no"
    return "open_ended"


def convert_sample(
    question: str,
    answer: str,
    image_filename: str,
    split: str,
    original_index: int,
) -> Optional[Dict[str, Any]]:
    """Converts a single VQA-RAD sample to pipeline format.

    Args:
        question: The question text.
        answer: The answer text.
        image_filename: Filename of the saved image.
        split: Dataset split (train/test).
        original_index: Original row index in the dataset.

    Returns:
        Converted sample dict, or None if invalid.
    """
    if not question or not question.strip():
        logger.warning(f"Skipping sample {original_index}: empty question")
        return None
    if not answer or not answer.strip():
        logger.warning(f"Skipping sample {original_index}: empty answer")
        return None

    return {
        "question": question.strip(),
        "answer": answer.strip(),
        "images": [image_filename],
        "question_type": classify_question_type(answer),
        "source": "vqa-rad",
        "split": split,
        "original_index": original_index,
    }


def convert_from_parquet(
    parquet_dir: str,
    output_dir: str,
    splits: Optional[List[str]] = None,
    force: bool = False,
):
    """Converts VQA-RAD Parquet files (with embedded images) to pipeline format.

    Extracts images from the HuggingFace Image column and saves them to disk.
    Outputs a JSONL file with image path references.

    Args:
        parquet_dir: Directory containing VQA-RAD Parquet files.
        output_dir: Output directory for converted.jsonl and images/.
        splits: List of splits to convert (default: all found).
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: 'pyarrow' is required for Parquet conversion.")
        print("Install it with: pip install pyarrow")
        return

    input_path = Path(parquet_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {parquet_dir}")

    parquet_files = sorted(input_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")

    # Filter by split if specified
    if splits:
        filtered = []
        for pf in parquet_files:
            for split_name in splits:
                if pf.stem.startswith(split_name) or pf.stem == split_name:
                    filtered.append((pf, split_name))
                    break
        if not filtered:
            available = [pf.name for pf in parquet_files]
            raise FileNotFoundError(
                f"No Parquet files matching splits {splits}. "
                f"Available files: {available}"
            )
    else:
        filtered = []
        for pf in parquet_files:
            split_name = pf.stem.split("-")[0] if "-" in pf.stem else pf.stem
            filtered.append((pf, split_name))

    logger.info(f"Found {len(filtered)} Parquet file(s) to convert")

    for parquet_path, split_name in filtered:
        split_dir, converted_file, images_dir = split_output_paths(output_dir, split_name)
        split_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        if should_skip_existing(converted_file, force=force):
            logger.info(f"Skipping split '{split_name}': output already exists at {converted_file}")
            continue

        converted_count = 0
        skipped_count = 0
        images_saved = 0
        global_index = 0
        image_hash_to_filename: Dict[int, str] = {}

        logger.info(f"Processing {parquet_path.name} (split={split_name})...")
        table = pq.read_table(str(parquet_path))
        columns = table.column_names
        logger.info(f"  Columns: {columns} ({len(table)} rows)")

        with open(converted_file, "w", encoding="utf-8") as outfile:
            for row_idx in range(len(table)):
                question = ""
                answer = ""
                image_filename = None

                for col_name in columns:
                    cell = table.column(col_name)[row_idx].as_py()

                    if col_name == "image" and cell is not None:
                        image_bytes = None
                        if isinstance(cell, dict):
                            image_bytes = cell.get("bytes")
                        elif isinstance(cell, bytes):
                            image_bytes = cell

                        if image_bytes:
                            content_hash = hash(image_bytes)
                            if content_hash in image_hash_to_filename:
                                image_filename = image_hash_to_filename[content_hash]
                            else:
                                image_filename = f"vqarad_{global_index:05d}.jpg"
                                dest = images_dir / image_filename
                                if not dest.exists():
                                    with open(dest, "wb") as img_file:
                                        img_file.write(image_bytes)
                                    images_saved += 1
                                image_hash_to_filename[content_hash] = image_filename

                    elif col_name == "question":
                        question = cell or ""
                    elif col_name == "answer":
                        answer = cell or ""

                if image_filename:
                    converted = convert_sample(
                        question, answer, image_filename,
                        split_name, global_index,
                    )
                    if converted:
                        outfile.write(json.dumps(converted, ensure_ascii=False) + "\n")
                        converted_count += 1
                    else:
                        skipped_count += 1
                else:
                    logger.warning(f"Skipping row {row_idx}: no image data")
                    skipped_count += 1

                global_index += 1
                if global_index % 500 == 0:
                    logger.info(
                        f"  [{split_name}] Progress: {global_index} rows | "
                        f"converted={converted_count} | images={images_saved}"
                    )

        logger.info(
            f"\nSplit '{split_name}' complete!\n"
            f"  Converted: {converted_count} samples\n"
            f"  Skipped:   {skipped_count} samples\n"
            f"  Images:    {images_saved} saved ({len(image_hash_to_filename)} unique)\n"
            f"  Output:    {converted_file}"
        )


def download_and_convert(
    output_dir: str = "data/vqa-rad/input",
    splits: Optional[List[str]] = None,
    force: bool = False,
):
    """Downloads VQA-RAD from HuggingFace and converts to pipeline format.

    Args:
        output_dir: Output directory.
        splits: List of splits to convert (default: ["test"]).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Please install it first:  pip install datasets")
        return

    if splits is None:
        splits = ["test"]

    print("=" * 60)
    print("Downloading VQA-RAD from HuggingFace...")
    print(f"Output directory: {Path(output_dir).resolve()}")
    print(f"Splits: {splits}")
    print("=" * 60)

    dataset = load_dataset("flaviagiammarino/vqa-rad")

    available_splits = list(dataset.keys())
    print(f"Available splits: {available_splits}")
    for split_name in available_splits:
        print(f"  [{split_name}] {len(dataset[split_name])} samples")

    for split_name in splits:
        if split_name not in dataset:
            print(f"WARNING: split '{split_name}' not found, skipping")
            continue

        split_dir, converted_file, images_dir = split_output_paths(output_dir, split_name)
        split_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        if should_skip_existing(converted_file, force=force):
            print(f"Skipping [{split_name}]: output already exists at {converted_file}")
            continue

        converted_count = 0
        skipped_count = 0
        images_saved = 0
        global_index = 0
        image_hash_to_filename: Dict[int, str] = {}
        split_data = dataset[split_name]
        print(f"\nConverting [{split_name}] ({len(split_data)} samples)...")

        with open(converted_file, "w", encoding="utf-8") as outfile:
            for row_idx in range(len(split_data)):
                item = split_data[row_idx]
                question = item.get("question", "")
                answer = item.get("answer", "")
                pil_image = item.get("image")

                image_filename = None
                if pil_image is not None:
                    image_bytes = pil_image.tobytes()
                    content_hash = hash(image_bytes)

                    if content_hash in image_hash_to_filename:
                        image_filename = image_hash_to_filename[content_hash]
                    else:
                        image_filename = f"vqarad_{global_index:05d}.jpg"
                        dest = images_dir / image_filename
                        if not dest.exists():
                            try:
                                if pil_image.mode not in ("RGB", "L"):
                                    pil_image = pil_image.convert("RGB")
                                pil_image.save(str(dest), format="JPEG", quality=95)
                                images_saved += 1
                            except Exception as error:
                                logger.warning(
                                    f"Failed to save image at index {global_index}: {error}"
                                )
                                image_filename = None
                        image_hash_to_filename[content_hash] = image_filename

                if image_filename:
                    converted = convert_sample(
                        question, answer, image_filename,
                        split_name, global_index,
                    )
                    if converted:
                        outfile.write(json.dumps(converted, ensure_ascii=False) + "\n")
                        converted_count += 1
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1

                global_index += 1
                if global_index % 500 == 0:
                    print(
                        f"  [{split_name}] Progress: {global_index} rows | "
                        f"converted={converted_count} | images={images_saved}"
                    )

        print(f"\n{'=' * 60}")
        print(f"Split '{split_name}' conversion complete!")
        print(f"  Converted: {converted_count} samples")
        print(f"  Skipped:   {skipped_count} samples")
        print(f"  Images:    {images_saved} saved ({len(image_hash_to_filename)} unique)")
        print(f"  Output:    {converted_file}")
        print("  Config:")
        print('     dataset: "vqa-rad"')
        print(f'     input_file: "{converted_file}"')
        print(f'     image_dir: "{images_dir}"')
        print(f'     split: "{split_name}"')
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert VQA-RAD to medhalu pipeline format"
    )
    parser.add_argument(
        "--from-parquet",
        type=str,
        default=None,
        metavar="DIR",
        help="Convert from local Parquet directory. "
             "If not provided, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/vqa-rad/input",
        help="Output directory (default: data/vqa-rad/input)",
    )
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=None,
        help="Split(s) to convert (default: test). "
             "Use 'all' for all splits.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if output files already exist.",
    )
    args = parser.parse_args()

    splits = args.split
    if splits and "all" in splits:
        splits = None
    elif splits is None:
        splits = ["test"]

    if args.from_parquet:
        convert_from_parquet(args.from_parquet, args.output_dir, splits, force=args.force)
    else:
        download_and_convert(args.output_dir, splits, force=args.force)


if __name__ == "__main__":
    main()
