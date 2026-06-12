import base64
import json
import logging
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import response

import yaml

try:
    from openai import BadRequestError, OpenAI
except ImportError:  # pragma: no cover
    BadRequestError = None
    OpenAI = None


logger = logging.getLogger(__name__)

REPLACE_EXPERIMENTS = (
    "replace_vr",
    "replace_kr",
    "replace_vr_kr",
)

REPLACE_DESCRIPTIONS = {
    "replace_vr": "Gold Visual Recognition + model continues",
    "replace_kr": "Gold Knowledge Recall + model continues",
    "replace_vr_kr": (
        "Gold Visual Recognition + Gold Knowledge Recall + "
        "model continues Reasoning"
    ),
}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _merge_dict_layers(*layers: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for layer in layers:
        if isinstance(layer, dict):
            _deep_merge_dict(merged, deepcopy(layer))
    return merged


def _apply_config_aliases(config: Dict[str, Any]) -> Dict[str, Any]:
    if "gold_filter" in config and "gold_cot_filter" not in config:
        config["gold_cot_filter"] = deepcopy(config["gold_filter"])
    if "replace" in config and "replace_experiment" not in config:
        config["replace_experiment"] = deepcopy(config["replace"])
    return config


def _registry_for_purpose(config: Dict[str, Any], purpose: str) -> Dict[str, Any]:
    if purpose == "judge":
        registry = config.get("judge_registry", {})
        if isinstance(registry, dict):
            return registry
    registry = config.get("model_registry", {})
    return registry if isinstance(registry, dict) else {}


def _resolve_model_spec(
    config: Dict[str, Any],
    model_ref: str,
    purpose: str = "default",
) -> Dict[str, Any]:
    registry = _registry_for_purpose(config, purpose)
    if isinstance(model_ref, str) and model_ref in registry:
        spec = deepcopy(registry[model_ref])
        spec.setdefault("model_name", model_ref)
        spec.setdefault("registry_key", model_ref)
        return spec
    if purpose != "default":
        fallback_registry = _registry_for_purpose(config, "default")
        if isinstance(model_ref, str) and model_ref in fallback_registry:
            spec = deepcopy(fallback_registry[model_ref])
            spec.setdefault("model_name", model_ref)
            spec.setdefault("registry_key", model_ref)
            return spec
    return {
        "model_name": model_ref,
        "registry_key": model_ref,
    }


def _resolve_model_name(
    config: Dict[str, Any],
    model_ref: str,
    purpose: str = "default",
) -> str:
    return _resolve_model_spec(
        config,
        model_ref,
        purpose=purpose,
    ).get("model_name", model_ref)


def _apply_dataset_selection(config: Dict[str, Any], dataset_key: str) -> None:
    datasets = config.get("datasets", {})
    if dataset_key not in datasets:
        raise KeyError(
            f"Unknown dataset key '{dataset_key}'. Available: {sorted(datasets)}"
        )

    dataset_spec = deepcopy(datasets[dataset_key])
    paths = config.setdefault("paths", {})
    paths["dataset"] = dataset_spec.get("dataset", dataset_key)
    paths["input_file"] = dataset_spec["input_file"]
    paths["image_dir"] = dataset_spec.get("image_dir")
    paths["split"] = dataset_spec.get("split")
    paths["dataset_key"] = dataset_key


def _apply_model_selection(
    config: Dict[str, Any],
    field_name: str,
    model_ref: str,
    allow_backend: bool = False,
    purpose: str = "default",
) -> Dict[str, Any]:
    model_spec = _resolve_model_spec(config, model_ref, purpose=purpose)
    config.setdefault("models", {})
    config["models"][field_name] = model_spec["model_name"]

    if allow_backend:
        api_override = deepcopy(model_spec.get("api"))
        if api_override:
            config.setdefault("api", {})
            config["api"]["model_cot"] = api_override

    overrides = model_spec.get("overrides")
    if isinstance(overrides, dict):
        _deep_merge_dict(config, overrides)

    return model_spec


def _normalize_default_model_refs(config: Dict[str, Any]) -> None:
    models = config.setdefault("models", {})

    gold_models = models.get("gold_cot")
    if isinstance(gold_models, str):
        gold_models = [gold_models]
    if isinstance(gold_models, list):
        models["gold_cot"] = [
            _resolve_model_name(config, model_ref)
            for model_ref in gold_models
        ]

    model_ref = models.get("model_cot")
    if isinstance(model_ref, str):
        model_spec = _resolve_model_spec(config, model_ref, purpose="model_cot")
        models["model_cot"] = model_spec["model_name"]
        api_override = deepcopy(model_spec.get("api"))
        if api_override:
            config.setdefault("api", {})
            config["api"]["model_cot"] = api_override

    judge_ref = models.get("judge")
    if isinstance(judge_ref, str):
        models["judge"] = _resolve_model_name(config, judge_ref, purpose="judge")

    report_ref = config.get("report", {}).get("model")
    if isinstance(report_ref, str):
        config.setdefault("report", {})
        config["report"]["model"] = _resolve_model_name(config, report_ref)

    gold_filter_config = config.get("gold_cot_filter", {})
    judge_ref = gold_filter_config.get("judge_model")
    if isinstance(judge_ref, str):
        config["gold_cot_filter"]["judge_model"] = _resolve_model_name(
            config,
            judge_ref,
            purpose="judge",
        )


def resolve_runtime_config(
    config: Dict[str, Any],
    dataset_key: Optional[str] = None,
    gold_model_key: Optional[str] = None,
    test_model_key: Optional[str] = None,
) -> Dict[str, Any]:
    resolved = deepcopy(config)
    _apply_config_aliases(resolved)

    if dataset_key:
        _apply_dataset_selection(resolved, dataset_key)

    if gold_model_key:
        resolved.setdefault("models", {})
        resolved["models"]["gold_cot"] = [
            _resolve_model_name(resolved, gold_model_key)
        ]

    if test_model_key:
        _apply_model_selection(
            resolved,
            field_name="model_cot",
            model_ref=test_model_key,
            allow_backend=True,
            purpose="model_cot",
        )

    _normalize_default_model_refs(resolved)

    experiment_meta = resolved.setdefault("experiment", {})
    if dataset_key:
        experiment_meta["dataset_key"] = dataset_key
    if gold_model_key:
        experiment_meta["gold_model_key"] = gold_model_key
    if test_model_key:
        experiment_meta["test_model_key"] = test_model_key

    return resolved


def load_config(
    config_path: str,
    dataset_key: Optional[str] = None,
    gold_model_key: Optional[str] = None,
    test_model_key: Optional[str] = None,
) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)
    return resolve_runtime_config(
        config,
        dataset_key=dataset_key,
        gold_model_key=gold_model_key,
        test_model_key=test_model_key,
    )


def load_matrix(matrix_path: str) -> Dict[str, List[str]]:
    path = Path(matrix_path)
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        with open(path, "r", encoding="utf-8") as file_handle:
            matrix = yaml.safe_load(file_handle) or {}
    else:
        matrix = {}
        with open(path, "r", encoding="utf-8") as file_handle:
            for raw_line in file_handle:
                line = raw_line.split("#", 1)[0].strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                items = [
                    item.strip()
                    for item in value.split(",")
                    if item.strip()
                ]
                matrix[key.strip()] = items

    normalized = {
        "gold_models": list(matrix.get("gold_models", [])),
        "test_models": list(matrix.get("test_models", [])),
        "datasets": list(matrix.get("datasets", [])),
    }

    if not normalized["gold_models"]:
        raise ValueError("Matrix must define at least one gold model.")
    if not normalized["test_models"]:
        raise ValueError("Matrix must define at least one test model.")
    if not normalized["datasets"]:
        raise ValueError("Matrix must define at least one dataset.")

    return normalized


def resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    data_dir = config["paths"].get("data_dir", "data")
    results_dir = config["paths"].get("results_dir", "results")
    dataset = config["paths"]["dataset"]
    dataset_key = config["paths"].get("dataset_key") or dataset
    results_dataset_dir = os.path.join(results_dir, dataset_key)

    gold_models = config["models"]["gold_cot"]
    if isinstance(gold_models, str):
        gold_models = [gold_models]
    gold_model_name = gold_models[0]
    model_name = config["models"]["model_cot"]

    gold_dir = os.path.join(results_dataset_dir, "gold", gold_model_name)
    eval_dir = os.path.join(results_dataset_dir, "eval", model_name)
    replace_dir = os.path.join(eval_dir, "replace")
    reports_dir = os.path.join(results_dataset_dir, "reports")

    replace_generation_files = {
        experiment: os.path.join(replace_dir, f"{experiment}.jsonl")
        for experiment in REPLACE_EXPERIMENTS
    }
    replace_judge_files = {
        experiment: os.path.join(replace_dir, f"{experiment}_judge.jsonl")
        for experiment in REPLACE_EXPERIMENTS
    }
    replace_step_hallucination_judge_files = {
        experiment: os.path.join(replace_dir, f"{experiment}_step_hallucination_judge.jsonl")
        for experiment in REPLACE_EXPERIMENTS
    }

    return {
        "data_dir": data_dir,
        "results_dir": results_dir,
        "dataset": dataset,
        "dataset_key": dataset_key,
        "gold_model": gold_model_name,
        "model_name": model_name,
        "results_dataset_dir": results_dataset_dir,
        "gold_dir": gold_dir,
        "eval_dir": eval_dir,
        "replace_dir": replace_dir,
        "image_reports_file": os.path.join(reports_dir, "image_reports.json"),
        "gold_cot_candidates_file": os.path.join(gold_dir, "candidates.jsonl"),
        "gold_cot_file": os.path.join(gold_dir, "gold_cot.jsonl"),
        "rejected_wrong_answer_file": os.path.join(
            gold_dir, "rejected_wrong_answer.jsonl"
        ),
        "rejected_invalid_cot_file": os.path.join(
            gold_dir, "rejected_invalid_cot.jsonl"
        ),
        "filter_stats_file": os.path.join(gold_dir, "filter_stats.json"),
        "model_cot_file": os.path.join(eval_dir, "model_cot.jsonl"),
        "judge_result_file": os.path.join(eval_dir, "judge.json"),
        "judge_details_file": os.path.join(eval_dir, "judge_details.jsonl"),
        "step_hallucination_judge_file": os.path.join(
            eval_dir, "step_hallucination_judge.jsonl"
        ),
        "replace_summary_file": os.path.join(replace_dir, "summary.json"),
        "replace_generation_files": replace_generation_files,
        "replace_judge_files": replace_judge_files,
        "replace_step_hallucination_judge_files": (
            replace_step_hallucination_judge_files
        ),
    }


def get_model_cot_api_config(config: Dict[str, Any]) -> Tuple[str, str]:
    model_cot_api = config.get("api", {}).get("model_cot", {})
    base_url = model_cot_api.get("base_url") or config["api"]["base_url"]
    api_key = (
        model_cot_api.get("api_key")
        or config["api"].get("api_key")
        or os.environ.get("OPENAI_API_KEY")
    )
    return base_url, api_key


def get_judge_api_config(config: Dict[str, Any]) -> Tuple[str, str]:
    judge_api = config.get("api", {}).get("judge", {})
    base_url = judge_api.get("base_url") or config["api"]["base_url"]
    api_key = (
        judge_api.get("api_key")
        or config["api"].get("api_key")
        or os.environ.get("OPENAI_API_KEY")
    )
    return base_url, api_key


def get_generation_request_config(config: Dict[str, Any]) -> Dict[str, Any]:
    generation = config.get("generation", {})
    return {
        "timeout": float(generation.get("timeout", 240.0)),
        "max_tokens": int(generation.get("max_tokens", 16384)),
        "temperature": float(generation.get("temperature", 0.2)),
        "top_p": float(generation.get("top_p", 0.2)),
        "retry_attempts": int(generation.get("retry_attempts", 50)),
    }


def get_judge_request_config(config: Dict[str, Any]) -> Dict[str, Any]:
    judge = config.get("judge", {})
    return {
        "timeout": float(judge.get("timeout", 240.0)),
        "max_tokens": int(judge.get("max_tokens", 16384)),
        "temperature": float(judge.get("temperature", 0.2)),
        "top_p": float(judge.get("top_p", 0.2)),
        "retry_attempts": int(judge.get("retry_attempts", 50)),
    }


def resolve_model_api_config(
    config: Dict[str, Any],
    model_ref: str,
    purpose: str = "default",
) -> Dict[str, str]:
    model_spec = _resolve_model_spec(config, model_ref, purpose=purpose)

    if purpose == "model_cot":
        base_url, api_key = get_model_cot_api_config(config)
    elif purpose == "judge":
        base_url, api_key = get_judge_api_config(config)
    else:
        base_url = config["api"]["base_url"]
        api_key = config["api"].get("api_key") or os.environ.get("OPENAI_API_KEY")

    api_override = model_spec.get("api")
    if isinstance(api_override, dict):
        base_url = api_override.get("base_url") or base_url
        api_key = api_override.get("api_key") or api_key

    return {
        "model_name": model_spec.get("model_name", model_ref),
        "base_url": base_url,
        "api_key": api_key,
    }


class CoTAPIWrapper:
    def __init__(
        self,
        model_path: str,
        base_url: str,
        api_key: str,
        max_tokens: int = 4096,
        timeout: float = 300.0,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
    ):
        if OpenAI is None:
            raise ImportError(
                "The 'openai' package is required for API-based generation."
            )
        self.model_path = model_path
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def generate(self, request_item: Dict[str, Any]) -> str:
        messages_data = request_item["messages"]
        system_prompt = messages_data.get("system", "")
        user_prompt = messages_data.get("prompt", "")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        image_list = messages_data.get("image_base64_list")
        if image_list:
            content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            for image_entry in image_list:
                mime_type = image_entry.get("mime", "image/png")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime_type};base64,{image_entry['base64']}"
                            ),
                        },
                    }
                )
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_prompt})

        request_kwargs: Dict[str, Any] = {
            "model": self.model_path,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        }
        if self.top_p is not None:
            request_kwargs["top_p"] = self.top_p

        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except BadRequestError:
            raise
        if isinstance(response, str):
            return response.strip()
        return response.choices[0].message.content.strip()


def create_model_cot_wrapper(
    config: Dict[str, Any],
    model_name: str,
    max_tokens: int,
):
    resolved = resolve_model_api_config(
        config=config,
        model_ref=model_name,
        purpose="model_cot",
    )
    request_config = get_generation_request_config(config)
    return CoTAPIWrapper(
        model_path=resolved["model_name"],
        base_url=resolved["base_url"],
        api_key=resolved["api_key"],
        max_tokens=int(request_config.get("max_tokens", max_tokens)),
        timeout=float(request_config.get("timeout", 240.0)),
        temperature=float(request_config.get("temperature", 0.2)),
        top_p=float(request_config.get("top_p", 0.2)),
    )


COT_STEPS = ["Visual Recognition", "Knowledge Recall", "Reasoning"]
COT_STEP_KEY_MAP = {
    "Visual Recognition": "visual_recognition",
    "Knowledge Recall": "knowledge_recall",
    "Reasoning": "reasoning",
}
COT_PARSE_PATTERN = re.compile(
    r"\[(" + "|".join(re.escape(step) for step in COT_STEPS) + r")\]\s*"
    r"(.*?)"
    r"(?=\[(?:" + "|".join(re.escape(step) for step in COT_STEPS) + r")\]|\Z)",
    re.DOTALL,
)
ANSWER_PATTERN = re.compile(r"\[Answer\]\s*(.+?)$", re.MULTILINE)


def _parse_answer(text: str) -> str:
    match = ANSWER_PATTERN.search(text)
    if not match:
        return ""
    return match.group(1).strip()


def _strip_answer_line(text: str) -> str:
    return ANSWER_PATTERN.sub("", text).strip()


def parse_cot(raw_cot: str) -> Optional[Dict[str, Any]]:
    matches = COT_PARSE_PATTERN.findall(raw_cot)
    if not matches:
        return None

    result = {}
    for step_name, content in matches:
        content = content.strip()
        if not content:
            continue
        step_key = COT_STEP_KEY_MAP.get(step_name, step_name.lower())
        step_data = {
            "text": _strip_answer_line(content)
            if step_key == "reasoning"
            else content,
        }
        if step_key == "reasoning":
            step_data["answer"] = _parse_answer(content)
        result[step_key] = step_data
    return result if result else None


def validate_cot(parsed_cot: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    if parsed_cot is None:
        return False, "Parsing returned None"

    required_keys = ["visual_recognition", "knowledge_recall", "reasoning"]
    display_names = {
        "visual_recognition": "Visual Recognition",
        "knowledge_recall": "Knowledge Recall",
        "reasoning": "Reasoning",
    }

    for step_key in required_keys:
        if step_key not in parsed_cot:
            return False, f"Missing required step: [{display_names[step_key]}]"
        text = parsed_cot[step_key].get("text", "")
        if len(text.split()) < 3:
            return (
                False,
                f"Step [{display_names[step_key]}] has insufficient text (< 3 words)",
            )

    reasoning_answer = parsed_cot.get("reasoning", {}).get("answer", "")
    if not reasoning_answer:
        return False, "Reasoning step has no [Answer]"

    return True, "OK"


def extract_step_text(cot_data: Dict[str, Any], step_name: str) -> str:
    step = cot_data.get(step_name, {})
    if isinstance(step, dict):
        return step.get("text", "")
    return ""


def extract_answer(cot_data: Dict[str, Any]) -> str:
    reasoning = cot_data.get("reasoning", {})
    if isinstance(reasoning, dict):
        return reasoning.get("answer", "")
    return ""


def normalize_answer(answer: str) -> str:
    normalized = answer.strip().lower()
    normalized = normalized.rstrip(".,;:!?")
    return normalized


def _bigram_similarity(string_a: str, string_b: str) -> float:
    if not string_a or not string_b:
        return 0.0
    if string_a == string_b:
        return 1.0

    bigrams_a = set(string_a[index:index + 2] for index in range(len(string_a) - 1))
    bigrams_b = set(string_b[index:index + 2] for index in range(len(string_b) - 1))
    if not bigrams_a or not bigrams_b:
        return 0.0

    overlap = len(bigrams_a & bigrams_b)
    return 2.0 * overlap / (len(bigrams_a) + len(bigrams_b))


def check_answer_match(
    predicted: str,
    ground_truth: str,
    threshold: float = 0.8,
) -> bool:
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    if not pred_norm or not gt_norm:
        return False
    if pred_norm == gt_norm:
        return True
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True
    return _bigram_similarity(pred_norm, gt_norm) >= threshold


MIME_TYPE_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
}


def encode_image_to_base64(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        logger.warning("Image not found: %s", image_path)
        return None
    except Exception as error:
        logger.warning("Failed to encode image %s: %s", image_path, error)
        return None


def get_mime_type(image_path: str) -> str:
    extension = Path(image_path).suffix.lower()
    return MIME_TYPE_MAP.get(extension, "image/png")


def load_input_data(
    input_file: str,
    split_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    data = []
    file_path = Path(input_file)

    if file_path.suffix == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as error:
                    logger.warning(
                        "Skipping invalid JSON at line %s: %s",
                        line_number,
                        error,
                    )
    elif file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as file_handle:
            loaded = json.load(file_handle)
        data = loaded if isinstance(loaded, list) else [loaded]
    else:
        raise ValueError(
            f"Unsupported file format: {file_path.suffix}. "
            "Supported: .json, .jsonl."
        )

    multi_image_count = 0
    single_image_count = 0
    no_image_count = 0
    max_images = 0
    for sample in data:
        image_list = sample.get("images", [])
        num_images = len(image_list) if isinstance(image_list, list) else 0
        if num_images > 1:
            multi_image_count += 1
            max_images = max(max_images, num_images)
        elif num_images == 1:
            single_image_count += 1
        else:
            no_image_count += 1

    logger.info(
        "Loaded %s samples from %s | Images: %s single, %s multi (max %s), %s text-only",
        len(data),
        input_file,
        single_image_count,
        multi_image_count,
        max_images,
        no_image_count,
    )

    if split_filter:
        requested_splits = {
            part.strip().lower()
            for part in str(split_filter).split(",")
            if part.strip()
        }
        if requested_splits:
            filtered = [
                sample
                for sample in data
                if str(sample.get("split", "")).strip().lower() in requested_splits
            ]
            logger.info(
                "Applied split filter %s: %s/%s samples kept",
                sorted(requested_splits),
                len(filtered),
                len(data),
            )
            data = filtered

    return data


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    path = Path(file_path)
    if not path.exists():
        return data

    with open(path, "r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as error:
                logger.warning(
                    "Skipping invalid JSON at line %s in %s: %s",
                    line_number,
                    file_path,
                    error,
                )
    return data


def get_sample_identifier(sample: Dict[str, Any]) -> Optional[str]:
    source_id = sample.get("source_id")
    if source_id is not None:
        source_text = str(source_id).strip()
        if source_text:
            return f"source_id:{source_text}"

    original_index = sample.get("original_index")
    if original_index is not None:
        index_text = str(original_index).strip()
        if index_text:
            return f"original_index:{index_text}"

    return None


def load_completed_sample_ids(output_file: str) -> set:
    completed = set()
    output_path = Path(output_file)
    if not output_path.exists():
        return completed

    missing_id_count = 0
    with open(output_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = get_sample_identifier(item)
            if sample_id is not None:
                completed.add(sample_id)
            else:
                missing_id_count += 1

    if missing_id_count:
        logger.warning(
            "%s rows in %s are missing source_id/original_index and cannot "
            "participate in ID-based resume tracking.",
            missing_id_count,
            output_file,
        )

    return completed


def load_questions_from_gold_cot(gold_cot_file: str) -> List[Dict[str, Any]]:
    samples = []
    with open(gold_cot_file, "r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                logger.warning(
                    "Skipping invalid JSON at line %s: %s",
                    line_number,
                    error,
                )
                continue

            sample = {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "source": item.get("source", ""),
                "split": item.get("split", ""),
                "original_index": item.get("original_index", line_number - 1),
            }
            if "images" in item:
                sample["images"] = item["images"]
            elif "image" in item:
                sample["images"] = [item["image"]]

            for extra_key in (
                "options",
                "question_type",
                "source_id",
                "answer_letter",
                "specialty",
                "body_system",
            ):
                if extra_key in item:
                    sample[extra_key] = item[extra_key]
            samples.append(sample)

    logger.info("Loaded %s questions from %s", len(samples), gold_cot_file)
    return samples


def align_samples(
    reference_data: List[Dict[str, Any]],
    target_data: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    reference_by_key = {}
    duplicate_reference_ids = 0
    missing_reference_ids = 0
    for item in reference_data:
        key = get_sample_identifier(item)
        if key is None:
            missing_reference_ids += 1
            continue
        if key in reference_by_key:
            duplicate_reference_ids += 1
        reference_by_key[key] = item

    aligned = []
    unmatched_count = 0
    missing_target_ids = 0
    for target_item in target_data:
        key = get_sample_identifier(target_item)
        if key is None:
            missing_target_ids += 1
            continue
        if key in reference_by_key:
            aligned.append((reference_by_key[key], target_item))
        else:
            unmatched_count += 1

    if duplicate_reference_ids:
        logger.warning(
            "%s duplicate sample IDs found in reference data; later rows overwrite earlier ones.",
            duplicate_reference_ids,
        )
    if missing_reference_ids:
        logger.warning(
            "%s reference samples are missing source_id/original_index and were skipped.",
            missing_reference_ids,
        )
    if missing_target_ids:
        logger.warning(
            "%s target samples are missing source_id/original_index and were skipped.",
            missing_target_ids,
        )
    if unmatched_count:
        logger.warning(
            "%s target samples could not be matched by sample ID.",
            unmatched_count,
        )
    logger.info("Aligned %s sample pairs", len(aligned))
    return aligned


def resolve_image_paths(sample: Dict[str, Any], image_dir: Optional[str]) -> List[str]:
    if not image_dir:
        return []

    filenames: List[str] = []
    images_field = sample.get("images")
    if isinstance(images_field, list) and images_field:
        for entry in images_field:
            if isinstance(entry, str) and entry:
                filenames.append(entry)
            elif isinstance(entry, dict):
                filenames.append(entry.get("image_path") or entry.get("path") or "")
    else:
        single = (
            sample.get("image")
            or sample.get("image_path")
            or sample.get("img_name")
            or sample.get("image_name")
            or ""
        )
        if single:
            filenames.append(single)

    resolved = []
    for filename in filenames:
        if not filename:
            continue
        full_path = os.path.join(image_dir, filename)
        if os.path.exists(full_path):
            resolved.append(full_path)
        else:
            logger.warning("Image not found: %s, skipping", full_path)
    return resolved


def build_options_text(sample: Dict[str, Any]) -> str:
    options = sample.get("options")
    if isinstance(options, list) and options:
        return "Options:\n" + "\n".join(options) + "\n\n"
    return ""


def format_question_with_options(sample: Dict[str, Any]) -> str:
    question = sample.get("question", "")
    options = sample.get("options")
    if isinstance(options, list) and options:
        return f"{question}\n\nOptions:\n" + "\n".join(options)
    return question


def get_replace_generation_file(paths: Dict[str, Any], experiment: str) -> str:
    return paths["replace_generation_files"][experiment]


def get_replace_judge_file(paths: Dict[str, Any], experiment: str) -> str:
    return paths["replace_judge_files"][experiment]


def get_replace_step_hallucination_judge_file(
    paths: Dict[str, Any],
    experiment: str,
) -> str:
    return paths["replace_step_hallucination_judge_files"][experiment]
