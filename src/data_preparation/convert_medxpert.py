"""
Convert MedXpertQA (MM subset) to the medhalu pipeline format.

MedXpertQA original format (from HuggingFace TsinghuaC3I/MedXpertQA):
    {
        "id": "MM-26",
        "question": "What is the most likely diagnosis?\nA. Pneumonia\n...",
        "options": {"A": "Pneumonia", "B": "Tuberculosis", ...},
        "label": "C",
        "images": ["path/to/image.jpg", ...],
        ...
    }

Converted format (compatible with medhalu pipeline):
    {
        "question": "What is the most likely diagnosis?\nA. Pneumonia\n...",
        "answer": "C. Lung cancer",
        "answer_letter": "C",
        "options": ["A. Pneumonia", "B. Tuberculosis", "C. Lung cancer", ...],
        "image": "path/to/image.jpg",
        "question_type": "multiple_choice",
        "source_id": "MM-26",
        ...
    }

Usage:
    # Download from HuggingFace and convert
    pip install datasets
    python scripts/convert_medxpert.py

    # Or convert from existing JSONL (already downloaded via prepare_medxpertqa.ipynb)
    python scripts/convert_medxpert.py --input path/to/medxpertqa_mm_input.jsonl

Output:
    data/medxpert/input/test/converted.jsonl
    data/medxpert/input/test/images/  (downloaded from HuggingFace)
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


def split_output_paths(output_dir: str, split: str = "test") -> tuple[Path, Path, Path]:
    """Returns (<split_dir>, <converted_file>, <images_dir>) for a dataset split."""
    split_dir = Path(output_dir) / split
    converted_file = split_dir / "converted.jsonl"
    images_dir = split_dir / "images"
    return split_dir, converted_file, images_dir


def should_skip_existing(converted_file: Path, force: bool = False) -> bool:
    """Returns True if a non-empty converted file already exists and force is off."""
    return (not force) and converted_file.exists() and converted_file.stat().st_size > 0


def convert_hf_sample(sample: Dict[str, Any], split: str = "test") -> Optional[Dict[str, Any]]:
    """Converts a single HuggingFace MedXpertQA sample to pipeline format.

    Handles the HuggingFace raw format where options is a dict like:
        {"A": "Pneumonia", "B": "Tuberculosis", ...}

    Args:
        sample: Raw sample dict from the dataset.
        split: Dataset split label (e.g., ``"test"``).
    """
    sample_id = sample.get("id", "")
    question_text = sample.get("question", "")
    raw_options = sample.get("options", {})
    label = sample.get("label", "")

    if not question_text or not label:
        logger.warning(f"Skipping sample {sample_id}: missing question or label")
        return None

    # Build formatted options list: ["A. Pneumonia", "B. Tuberculosis", ...]
    if isinstance(raw_options, dict):
        formatted_options = [f"{letter}. {content}" for letter, content in sorted(raw_options.items())]
        answer_content = raw_options.get(label, "")
    elif isinstance(raw_options, list) and raw_options and isinstance(raw_options[0], dict):
        # Already converted format: [{"letter": "A", "content": "Pneumonia"}, ...]
        formatted_options = [f"{opt['letter']}. {opt['content']}" for opt in raw_options]
        answer_content = next(
            (opt["content"] for opt in raw_options if opt["letter"] == label),
            "",
        )
    elif isinstance(raw_options, list) and raw_options and isinstance(raw_options[0], str):
        # Simple list format: ["Pneumonia", "Tuberculosis", ...]
        formatted_options = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(raw_options)]
        label_index = ord(label) - 65 if len(label) == 1 and label.isalpha() else -1
        answer_content = raw_options[label_index] if 0 <= label_index < len(raw_options) else ""
    else:
        logger.warning(f"Skipping sample {sample_id}: unrecognized options format")
        return None

    # Compose full answer: "C. Lung cancer"
    full_answer = f"{label}. {answer_content}" if answer_content else label

    # Handle images: preserve ALL images in order for multi-image support
    raw_images = sample.get("images", [])
    image_paths = []
    for img_entry in raw_images:
        if isinstance(img_entry, dict):
            path = img_entry.get("image_path") or img_entry.get("path") or ""
            if path:
                image_paths.append(path)
        elif isinstance(img_entry, str) and img_entry:
            image_paths.append(img_entry)

    converted = {
        "question": question_text,
        "answer": full_answer,
        "answer_letter": label,
        "options": formatted_options,
        "question_type": "multiple_choice",
        "source_id": sample_id,
        "split": split,
    }

    if image_paths:
        converted["images"] = image_paths

    # Preserve extra metadata fields
    for extra_key in ["specialty", "body_system", "task", "source"]:
        if extra_key in sample:
            converted[extra_key] = sample[extra_key]

    return converted


def convert_from_jsonl(input_file: str, output_file: str, force: bool = False):
    """Converts an existing MedXpertQA JSONL file to pipeline format."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if should_skip_existing(output_path, force=force):
        logger.info(f"Skipping conversion: output already exists at {output_path}")
        return

    converted_count = 0
    skipped_count = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line_number, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                logger.warning(f"Skipping invalid JSON at line {line_number}: {error}")
                skipped_count += 1
                continue

            converted = convert_hf_sample(sample)
            if converted:
                outfile.write(json.dumps(converted, ensure_ascii=False) + "\n")
                converted_count += 1
            else:
                skipped_count += 1

    logger.info(f"Converted {converted_count} samples, skipped {skipped_count}")
    logger.info(f"Output saved to: {output_path}")


def _download_images_from_hf_repo(images_dir: Path):
    """Downloads the images.zip from MedXpertQA HuggingFace repo and extracts it.

    MedXpertQA stores all MM images in a single 'images.zip' file in the
    HuggingFace dataset repo.
    """
    import shutil
    import zipfile

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: 'huggingface_hub' library not installed.")
        print("Please install it:  pip install huggingface_hub")
        return

    # Check if images already exist
    existing_images = list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.jpg")) + \
                      list(images_dir.glob("*.JPG")) + list(images_dir.glob("*.png"))
    if len(existing_images) > 100:
        print(f"  Found {len(existing_images)} images already in {images_dir}, skipping download.")
        return

    print("  Downloading images.zip from HuggingFace repo...")
    try:
        zip_path = hf_hub_download(
            repo_id="TsinghuaC3I/MedXpertQA",
            filename="images.zip",
            repo_type="dataset",
        )
        print(f"  Downloaded to: {zip_path}")
    except Exception as error:
        logger.error(f"Failed to download images.zip: {error}")
        print("\n  If the download fails, you can manually download images.zip from:")
        print("  https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA/tree/main/MM")
        print(f"  Then extract it to: {images_dir}")
        return

    # Extract images
    print("  Extracting images...")
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        members = zip_file.namelist()
        print(f"  Found {len(members)} files in zip")
        for member in members:
            # Extract image files to the flat images directory
            filename = Path(member).name
            if not filename or filename.startswith("."):
                continue
            source = zip_file.open(member)
            dest_path = images_dir / filename
            with open(dest_path, "wb") as dest_file:
                shutil.copyfileobj(source, dest_file)

    final_count = len(list(images_dir.glob("*")))
    print(f"  Extracted {final_count} images to {images_dir}")


def download_and_convert(output_dir: str = "data/medxpert/input", force: bool = False):
    """Downloads MedXpertQA MM from HuggingFace and converts to pipeline format."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Please install it first:  pip install datasets")
        return

    split_dir, converted_file, images_dir = split_output_paths(output_dir, "test")
    split_dir.mkdir(parents=True, exist_ok=True)
    if should_skip_existing(converted_file, force=force):
        print(f"Skipping [test]: output already exists at {converted_file}")
        return

    print("=" * 60)
    print("Downloading MedXpertQA MM from HuggingFace...")
    print(f"Output directory: {split_dir.resolve()}")
    print("=" * 60)

    dataset = load_dataset("TsinghuaC3I/MedXpertQA", "MM")
    test_data = dataset["test"]

    print(f"\nLoaded {len(test_data)} samples")
    print(f"Columns: {test_data.column_names}")

    # Prepare images directory
    images_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image filenames and convert data
    converted_count = 0
    all_image_filenames = []

    with open(converted_file, "w", encoding="utf-8") as outfile:
        for index, item in enumerate(test_data):
            images_list = item.get("images", [])

            # Collect image filenames for batch download
            for image_ref in images_list:
                if isinstance(image_ref, str):
                    all_image_filenames.append(image_ref)

            # Build sample dict for conversion
            sample = dict(item)
            sample["images"] = images_list

            converted = convert_hf_sample(sample)
            if converted:
                outfile.write(json.dumps(converted, ensure_ascii=False) + "\n")
                converted_count += 1

            if (index + 1) % 100 == 0:
                print(f"  Processed {index + 1}/{len(test_data)} samples...")

    print(f"\nConverted {converted_count} samples")
    print(f"Output: {converted_file}")

    # Download images from HuggingFace repo (stored as images.zip)
    if all_image_filenames:
        print(f"\nFound {len(set(all_image_filenames))} unique image references, downloading...")
        _download_images_from_hf_repo(images_dir)
    else:
        print("\nNo images found in dataset.")

    print(f"Images: {images_dir}")
    print()
    print("=" * 60)
    print("Next steps:")
    print("  1. Update configs/config.yaml:")
    print('     dataset: "medxpert"')
    print('     input_file: "data/medxpert/input/test/converted.jsonl"')
    print('     image_dir: "data/medxpert/input/test/images"')
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
        description="Convert MedXpertQA to medhalu pipeline format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to existing MedXpertQA JSONL file. "
             "If not provided, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/medxpert/input",
        help="Output directory (default: data/medxpert/input)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output JSONL file path (default: {output_dir}/converted.jsonl)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if output files already exist.",
    )
    args = parser.parse_args()

    if args.input:
        output_file = args.output_file or str(split_output_paths(args.output_dir, "test")[1])
        convert_from_jsonl(args.input, output_file, force=args.force)
    else:
        download_and_convert(args.output_dir, force=args.force)


if __name__ == "__main__":
    main()
