import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"

REPLACE_EXPERIMENTS = ("replace_vr", "replace_kr", "replace_vr_kr")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate judge/replace/step-hallucination summaries into "
            "comparison tables."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Results root to scan. Default: {DEFAULT_RESULTS_ROOT}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset directory filter, e.g. vqa-rad-test.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model directory names to include.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("model", "acc"),
        default="acc",
        help="Sort rows by model name or baseline answer accuracy.",
    )
    parser.add_argument(
        "--table",
        choices=("all", "answer", "hallucination", "acc_delta", "hall_base", "hall_delta"),
        default="all",
        help="Which table(s) to print.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown output path. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional markdown output path. Overrides --output when both are set.",
    )
    parser.add_argument(
        "--output-csv-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for CSV exports. Writes one CSV per table, "
            "for example answer_accuracy_table.csv."
        ),
    )
    return parser.parse_args()


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def format_delta(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def format_count(numerator: Optional[int], denominator: Optional[int]) -> str:
    if numerator is None or denominator is None:
        return "-"
    return f"{numerator}/{denominator}"


def step_rate(summary: Optional[Dict[str, Any]], step_name: str) -> Optional[float]:
    if not summary:
        return None
    steps = summary.get("steps", {})
    if not isinstance(steps, dict):
        return None
    step_info = steps.get(step_name)
    if not isinstance(step_info, dict):
        return None
    value = step_info.get("hallucination_rate")
    return float(value) if isinstance(value, (int, float)) else None


def build_paper_step_summary(
    step_model: Optional[Dict[str, Any]],
    step_replace: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    # Paper-style decomposition reads each component from the corresponding
    # replace-based step hallucination summary.
    return {
        "steps": {
            "visual_recognition": {
                "hallucination_rate": step_rate(
                    step_replace.get("replace_kr"),
                    "visual_recognition",
                ),
            },
            "knowledge_recall": {
                "hallucination_rate": step_rate(
                    step_replace.get("replace_vr"),
                    "knowledge_recall",
                ),
            },
            "reasoning": {
                "hallucination_rate": step_rate(
                    step_replace.get("replace_vr_kr"),
                    "reasoning",
                ),
            },
        }
    }


def reasoning_delta(
    step_model: Optional[Dict[str, Any]],
    step_replace_summary: Optional[Dict[str, Any]],
) -> Optional[float]:
    baseline = step_rate(step_model, "reasoning")
    replaced = step_rate(step_replace_summary, "reasoning")
    if baseline is None or replaced is None:
        return None
    return replaced - baseline


def make_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "_No rows found._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_csv_table(path: Path, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def iter_eval_dirs(
    results_root: Path,
    dataset_filter: Optional[str],
    model_filters: Optional[Sequence[str]],
) -> Iterable[Path]:
    model_filter_set = set(model_filters or [])
    datasets = [results_root / dataset_filter] if dataset_filter else sorted(results_root.iterdir())
    for dataset_dir in datasets:
        eval_root = dataset_dir / "eval"
        if not eval_root.is_dir():
            continue
        for model_dir in sorted(eval_root.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_filter_set and model_dir.name not in model_filter_set:
                continue
            yield model_dir


def collect_rows(
    results_root: Path,
    dataset_filter: Optional[str],
    model_filters: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_dir in iter_eval_dirs(results_root, dataset_filter, model_filters):
        dataset = model_dir.parent.parent.name
        model = model_dir.name

        judge = safe_load_json(model_dir / "judge.json")
        replace = safe_load_json(model_dir / "replace" / "summary.json")
        step_model = safe_load_json(model_dir / "step_hallucination_judge_summary.json")
        step_replace = {
            experiment: safe_load_json(
                model_dir / "replace" / f"{experiment}_step_hallucination_judge_summary.json"
            )
            for experiment in REPLACE_EXPERIMENTS
        }
        step_paper = build_paper_step_summary(step_model, step_replace)

        judge_config = judge.get("config", {}) if judge else {}
        aggregated = judge.get("aggregated_metrics", {}) if judge else {}
        answer_accuracy = aggregated.get("answer_accuracy", {}) if isinstance(aggregated, dict) else {}
        replace_experiments = replace.get("experiments", {}) if replace else {}

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "gold_model": judge_config.get("gold_model"),
                "judge_model": judge_config.get("judge_model"),
                "baseline_acc": answer_accuracy.get("acc"),
                "baseline_correct": answer_accuracy.get("correct_count"),
                "baseline_total": answer_accuracy.get("total"),
                "replace": replace_experiments,
                "step_model": step_model,
                "step_paper": step_paper,
                "step_replace": step_replace,
            }
        )
    return rows


def row_sort_key(row: Dict[str, Any], sort_by: str):
    if sort_by == "model":
        return (row["dataset"], row["model"].lower())
    acc = row.get("baseline_acc")
    acc_key = -1.0 if not isinstance(acc, (int, float)) else -float(acc)
    return (row["dataset"], acc_key, row["model"].lower())


def answer_table_data(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    headers = [
        "dataset",
        "model",
        "ACC",
        "ΔV",
        "ΔK",
        "ΔVK",
    ]
    body: List[List[str]] = []
    for row in rows:
        replace = row.get("replace", {})
        vr = replace.get("replace_vr", {}) if isinstance(replace, dict) else {}
        kr = replace.get("replace_kr", {}) if isinstance(replace, dict) else {}
        vr_kr = replace.get("replace_vr_kr", {}) if isinstance(replace, dict) else {}
        body.append(
            [
                str(row["dataset"]),
                str(row["model"]),
                format_metric(row.get("baseline_acc")),
                format_delta(vr.get("delta_vs_baseline") if isinstance(vr, dict) else None),
                format_delta(kr.get("delta_vs_baseline") if isinstance(kr, dict) else None),
                format_delta(vr_kr.get("delta_vs_baseline") if isinstance(vr_kr, dict) else None),
            ]
        )
    return headers, body


def build_answer_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers, body = answer_table_data(rows)
    return make_markdown_table(headers, body)


def hallucination_table_data(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    headers = [
        "dataset",
        "model",
        "H^V",
        "H^K",
        "H^R",
    ]
    body: List[List[str]] = []
    for row in rows:
        body.append(
            [
                str(row["dataset"]),
                str(row["model"]),
                format_metric(step_rate(row.get("step_paper"), "visual_recognition")),
                format_metric(step_rate(row.get("step_paper"), "knowledge_recall")),
                format_metric(step_rate(row.get("step_paper"), "reasoning")),
            ]
        )
    return headers, body


def build_hallucination_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers, body = hallucination_table_data(rows)
    return make_markdown_table(headers, body)


def hallucination_delta_table_data(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
    headers = [
        "dataset",
        "model",
        "Rep-V",
        "Rep-K",
        "Rep-VK",
    ]
    body: List[List[str]] = []
    for row in rows:
        step_model = row.get("step_model")
        step_replace = row.get("step_replace", {})
        rv = step_replace.get("replace_vr") if isinstance(step_replace, dict) else None
        rk = step_replace.get("replace_kr") if isinstance(step_replace, dict) else None
        rvkr = step_replace.get("replace_vr_kr") if isinstance(step_replace, dict) else None
        body.append(
            [
                str(row["dataset"]),
                str(row["model"]),
                format_delta(reasoning_delta(step_model, rv)),
                format_delta(reasoning_delta(step_model, rk)),
                format_delta(reasoning_delta(step_model, rvkr)),
            ]
        )
    return headers, body


def build_hallucination_delta_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers, body = hallucination_delta_table_data(rows)
    return make_markdown_table(headers, body)


def render_report(rows: Sequence[Dict[str, Any]], table: str) -> str:
    sections: List[str] = []
    if table in {"all", "answer", "acc_delta"}:
        sections.append("## Answer Accuracy Delta Table")
        sections.append(build_answer_table(rows))
    if table in {"all", "hallucination", "hall_base"}:
        sections.append("## Step Hallucination Base Table")
        sections.append(build_hallucination_table(rows))
    if table == "hall_delta":
        sections.append("## Step Hallucination Delta Table")
        sections.append(build_hallucination_delta_table(rows))
    return "\n\n".join(sections).strip() + "\n"


def export_csv_tables(rows: Sequence[Dict[str, Any]], table: str, output_dir: Path) -> List[Path]:
    written: List[Path] = []
    if table in {"all", "answer", "acc_delta"}:
        headers, body = answer_table_data(rows)
        answer_path = output_dir / "answer_accuracy_delta_table.csv"
        write_csv_table(answer_path, headers, body)
        written.append(answer_path)
    if table in {"all", "hallucination", "hall_base"}:
        headers, body = hallucination_table_data(rows)
        hall_base_path = output_dir / "step_hallucination_base_table.csv"
        write_csv_table(hall_base_path, headers, body)
        written.append(hall_base_path)
    if table == "hall_delta":
        headers, body = hallucination_delta_table_data(rows)
        hall_delta_path = output_dir / "step_hallucination_delta_table.csv"
        write_csv_table(hall_delta_path, headers, body)
        written.append(hall_delta_path)
    return written


def main() -> int:
    args = parse_arguments()
    rows = collect_rows(
        results_root=args.results_root,
        dataset_filter=args.dataset,
        model_filters=args.models,
    )
    rows = sorted(rows, key=lambda row: row_sort_key(row, args.sort_by))
    report = render_report(rows, args.table)
    markdown_output = args.output_md or args.output

    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(report, encoding="utf-8")
        print(f"Saved markdown table report to: {markdown_output}")
    if args.output_csv_dir is not None:
        written_paths = export_csv_tables(rows, args.table, args.output_csv_dir)
        for path in written_paths:
            print(f"Saved CSV table to: {path}")
    if markdown_output is None and args.output_csv_dir is None:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
