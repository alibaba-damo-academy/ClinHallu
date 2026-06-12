import argparse
import subprocess
import sys
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run replace generation and/or replace judging"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["generate", "judge", "all"],
        help="Which part of the replace workflow to run.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="replace_all",
        choices=["replace_all", "replace_vr", "replace_kr", "replace_vr_kr"],
        help="Which replace variant(s) to run.",
    )
    parser.add_argument("--dataset-key", type=str, default=None,
                        help="Dataset key from the unified config registry.")
    parser.add_argument("--gold-model", type=str, default=None,
                        help="Gold model key/name override.")
    parser.add_argument("--test-model", type=str, default=None,
                        help="Test model key/name override.")
    return parser.parse_args()


def run_command(command):
    subprocess.run(command, check=True)


def main():
    args = parse_arguments()
    src_root = Path(__file__).resolve().parents[1]
    python_bin = sys.executable

    if args.stage in {"generate", "all"}:
        run_command(
            [
                python_bin,
                str(src_root / "model_evaluation" / "01_generate_cot.py"),
                "--config",
                args.config,
                "--mode",
                args.mode,
                *(
                    ["--dataset-key", args.dataset_key]
                    if args.dataset_key
                    else []
                ),
                *(
                    ["--gold-model", args.gold_model]
                    if args.gold_model
                    else []
                ),
                *(
                    ["--test-model", args.test_model]
                    if args.test_model
                    else []
                ),
            ]
        )

    if args.stage in {"judge", "all"}:
        run_command(
            [
                python_bin,
                str(src_root / "model_evaluation" / "02_judge_answers.py"),
                "--config",
                args.config,
                "--target",
                args.mode,
                *(
                    ["--dataset-key", args.dataset_key]
                    if args.dataset_key
                    else []
                ),
                *(
                    ["--gold-model", args.gold_model]
                    if args.gold_model
                    else []
                ),
                *(
                    ["--test-model", args.test_model]
                    if args.test_model
                    else []
                ),
            ]
)


if __name__ == "__main__":
    main()
