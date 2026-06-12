"""
Convert PathVQA dataset to the medhalu pipeline format.

PathVQA is a pathology visual question answering dataset with ~32k QA pairs
on ~5k pathology images. Questions are either open-ended or binary yes/no.

PathVQA original format (from HuggingFace flaviagiammarino/path-vqa):
    {
        "image": <PIL Image>,
        "question": "where are liver stem cells (oval cells) located?",
        "answer": "in the canals of hering"
    }

Converted format (compatible with medhalu pipeline):
    {
        "question": "where are liver stem cells (oval cells) located?",
        "answer": "in the canals of hering",
        "images": ["pathvqa_00001.jpg"],
        "question_type": "open_ended" | "yes_no",
        "source": "pathvqa",
        "split": "test",
        "original_index": 0
    }

Three conversion modes:
    1. Download from HuggingFace → JSONL + extracted images (default)
    2. From local Parquet files → JSONL + extracted images (--from-parquet)
    3. Keep Parquet as-is (existing download_pathvqa.py approach)

Usage:
    # Download from HuggingFace and convert
    python src/data_preparation/convert_pathvqa.py

    # Convert from already-downloaded Parquet files
    python src/data_preparation/convert_pathvqa.py --from-parquet data/path-vqa/input/data

    # Specify split and output directory
    python src/data_preparation/convert_pathvqa.py --split test --output-dir data/pathvqa/input

Output:
    data/pathvqa/input/<split>/converted.jsonl
    data/pathvqa/input/<split>/images/
"""

import json
import logging
import os
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    """Converts a single PathVQA sample to pipeline format.

    Args:
        question: The question text.
        answer: The answer text.
        image_filename: Filename of the saved image.
        split: Dataset split (train/validation/test).
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
        "source": "pathvqa",
        "split": split,
        "original_index": original_index,
    }


def _question_image_key(sample: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    question = str(sample.get("question", "")).strip().lower()
    images = sample.get("images")
    if isinstance(images, list):
        image_key = tuple(str(image_name) for image_name in images)
    elif images:
        image_key = (str(images),)
    else:
        image_key = tuple()
    return question, image_key


def _normalize_answer(answer: Any) -> str:
    return str(answer or "").strip().lower()


def filter_question_image_conflicts(
    samples: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Deduplicates exact question-image-answer repeats and drops ambiguous groups."""
    grouped: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[_question_image_key(sample)].append(sample)

    filtered: List[Dict[str, Any]] = []
    stats = {
        "same_answer_groups": 0,
        "same_answer_rows": 0,
        "same_answer_dropped": 0,
        "diff_answer_groups": 0,
        "diff_answer_rows": 0,
        "diff_answer_dropped": 0,
    }

    for group in grouped.values():
        if len(group) == 1:
            filtered.append(group[0])
            continue

        distinct_answers = {_normalize_answer(sample.get("answer")) for sample in group}
        if len(distinct_answers) == 1:
            filtered.append(group[0])
            stats["same_answer_groups"] += 1
            stats["same_answer_rows"] += len(group)
            stats["same_answer_dropped"] += len(group) - 1
            continue

        stats["diff_answer_groups"] += 1
        stats["diff_answer_rows"] += len(group)
        stats["diff_answer_dropped"] += len(group)

    return filtered, stats


def write_converted_samples(
    converted_file: Path,
    samples: Sequence[Dict[str, Any]],
) -> None:
    with open(converted_file, "w", encoding="utf-8") as outfile:
        for sample in samples:
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")


def convert_from_parquet(
    parquet_dir: str,
    output_dir: str,
    splits: Optional[List[str]] = None,
    force: bool = False,
    dedup_question_image: bool = True,
):
    """Converts PathVQA Parquet files (with embedded images) to pipeline format.

    Extracts images from the HuggingFace Image column and saves them to disk.
    Outputs a JSONL file with image path references.

    Args:
        parquet_dir: Directory containing PathVQA Parquet files.
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
    # Find Parquet files
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
        # Infer split from filename
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

        converted_candidates: List[Dict[str, Any]] = []
        skipped_count = 0
        images_saved = 0
        global_index = 0
        image_hash_to_filename: Dict[int, str] = {}

        logger.info(f"Processing {parquet_path.name} (split={split_name})...")
        table = pq.read_table(str(parquet_path))
        columns = table.column_names
        logger.info(f"  Columns: {columns} ({len(table)} rows)")

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
                            image_filename = f"pathvqa_{global_index:05d}.jpg"
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
                    converted_candidates.append(converted)
                else:
                    skipped_count += 1
            else:
                logger.warning(f"Skipping row {row_idx}: no image data")
                skipped_count += 1

            global_index += 1
            if global_index % 1000 == 0:
                logger.info(
                    f"  [{split_name}] Progress: {global_index} rows | "
                    f"prepared={len(converted_candidates)} | images={images_saved}"
                )

        conflict_stats = None
        converted_samples = converted_candidates
        if dedup_question_image:
            converted_samples, conflict_stats = filter_question_image_conflicts(
                converted_candidates
            )

        write_converted_samples(converted_file, converted_samples)

        logger.info(
            f"\nSplit '{split_name}' complete!\n"
            f"  Prepared:  {len(converted_candidates)} samples\n"
            f"  Converted: {len(converted_samples)} samples\n"
            f"  Skipped:   {skipped_count} samples\n"
            f"  Images:    {images_saved} saved ({len(image_hash_to_filename)} unique)\n"
            f"  Output:    {converted_file}"
        )
        if conflict_stats is not None:
            logger.info(
                "  Question+image filter: dropped %s exact duplicates across %s groups; "
                "dropped %s ambiguous rows across %s groups",
                conflict_stats["same_answer_dropped"],
                conflict_stats["same_answer_groups"],
                conflict_stats["diff_answer_dropped"],
                conflict_stats["diff_answer_groups"],
            )


def download_and_convert(
    output_dir: str = "data/pathvqa/input",
    splits: Optional[List[str]] = None,
    force: bool = False,
    dedup_question_image: bool = True,
):
    """Downloads PathVQA from HuggingFace and converts to pipeline format.

    Downloads the dataset, extracts images to disk, and converts QA pairs
    to pipeline-compatible JSONL format.

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
    print("Downloading PathVQA from HuggingFace...")
    print(f"Output directory: {Path(output_dir).resolve()}")
    print(f"Splits: {splits}")
    print("=" * 60)

    dataset = load_dataset("flaviagiammarino/path-vqa")

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

        converted_candidates: List[Dict[str, Any]] = []
        skipped_count = 0
        images_saved = 0
        global_index = 0
        image_hash_to_filename: Dict[int, str] = {}
        split_data = dataset[split_name]
        print(f"\nConverting [{split_name}] ({len(split_data)} samples)...")

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
                    image_filename = f"pathvqa_{global_index:05d}.jpg"
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
                    converted_candidates.append(converted)
                else:
                    skipped_count += 1
            else:
                skipped_count += 1

            global_index += 1
            if global_index % 1000 == 0:
                print(
                    f"  [{split_name}] Progress: {global_index} rows | "
                    f"prepared={len(converted_candidates)} | images={images_saved}"
                )

        conflict_stats = None
        converted_samples = converted_candidates
        if dedup_question_image:
            converted_samples, conflict_stats = filter_question_image_conflicts(
                converted_candidates
            )

        write_converted_samples(converted_file, converted_samples)

        print(f"\n{'=' * 60}")
        print(f"Split '{split_name}' conversion complete!")
        print(f"  Prepared:  {len(converted_candidates)} samples")
        print(f"  Converted: {len(converted_samples)} samples")
        print(f"  Skipped:   {skipped_count} samples")
        print(f"  Images:    {images_saved} saved ({len(image_hash_to_filename)} unique)")
        if conflict_stats is not None:
            print(
                "  Question+image filter: "
                f"dropped {conflict_stats['same_answer_dropped']} exact duplicates "
                f"across {conflict_stats['same_answer_groups']} groups; "
                f"dropped {conflict_stats['diff_answer_dropped']} ambiguous rows "
                f"across {conflict_stats['diff_answer_groups']} groups"
            )
        print(f"  Output:    {converted_file}")
        print("  Config:")
        print('     dataset: "pathvqa"')
        print(f'     input_file: "{converted_file}"')
        print(f'     image_dir: "{images_dir}"')
        print(f'     split: "{split_name}"')
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PathVQA to medhalu pipeline format"
    )
    parser.add_argument(
        "--from-parquet",
        type=str,
        default=None,
        metavar="DIR",
        help="Convert from local Parquet directory (e.g. data/path-vqa/input/data). "
             "If not provided, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pathvqa/input",
        help="Output directory (default: data/pathvqa/input)",
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
    parser.add_argument(
        "--keep-question-image-conflicts",
        action="store_true",
        help=(
            "Keep PathVQA rows that share the same question and image. "
            "By default, exact duplicates are collapsed and conflicting-answer "
            "groups are removed."
        ),
    )
    args = parser.parse_args()

    # Resolve splits
    splits = args.split
    if splits and "all" in splits:
        splits = None  # None means all splits
    elif splits is None:
        splits = ["test"]

    if args.from_parquet:
        convert_from_parquet(
            args.from_parquet,
            args.output_dir,
            splits,
            force=args.force,
            dedup_question_image=not args.keep_question_image_conflicts,
        )
    else:
        download_and_convert(
            args.output_dir,
            splits,
            force=args.force,
            dedup_question_image=not args.keep_question_image_conflicts,
        )


if __name__ == "__main__":
    main()
