import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
for candidate in (str(SRC_ROOT), str(PROJECT_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from common import CoTAPIWrapper, encode_image_to_base64, get_mime_type, resolve_image_paths
from prompts.step_hallucination_judge import (
    KR_HALLUCINATION_JUDGE_USER_PROMPT_TEMPLATE,
    REASONING_HALLUCINATION_JUDGE_USER_PROMPT_TEMPLATE,
    STEP_HALLUCINATION_JUDGE_SYSTEM_PROMPT,
    VR_HALLUCINATION_JUDGE_USER_PROMPT_TEMPLATE,
)


logger = logging.getLogger(__name__)

STEP_PROMPTS = {
    "visual_recognition": VR_HALLUCINATION_JUDGE_USER_PROMPT_TEMPLATE,
    "knowledge_recall": KR_HALLUCINATION_JUDGE_USER_PROMPT_TEMPLATE,
    "reasoning": REASONING_HALLUCINATION_JUDGE_USER_PROMPT_TEMPLATE,
}

DEFAULT_STEP_HALLUCINATION_RESULT = {
    "hallucinated": 1,
    "reason": "Judge failed to produce a valid result",
}


def parse_step_hallucination_response(raw_response: str) -> Optional[Dict[str, Any]]:
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as error:
        logger.warning("Failed to parse step hallucination judge JSON: %s", error)
        return None

    hallucinated = result.get("hallucinated")
    reason = str(result.get("reason", "")).strip()

    try:
        hallucinated_int = int(hallucinated)
    except (TypeError, ValueError):
        logger.warning("Invalid step hallucination value: %s", hallucinated)
        return None
    if hallucinated_int not in {0, 1}:
        logger.warning("Step hallucination value must be 0 or 1: %s", hallucinated_int)
        return None
    if not reason:
        logger.warning("Missing step hallucination reason")
        return None

    return {
        "hallucinated": hallucinated_int,
        "reason": reason,
    }


def _build_support_context(
    reference_sample: Dict[str, Any],
    candidate_sample: Dict[str, Any],
) -> str:
    lines = []
    ref_cot = reference_sample.get("gold_cot", {})
    candidate_cot = candidate_sample.get("cot", {})

    reference_vr = (
        ref_cot.get("visual_recognition", {}).get("text", "")
        if isinstance(ref_cot, dict)
        else ""
    )
    reference_kr = (
        ref_cot.get("knowledge_recall", {}).get("text", "")
        if isinstance(ref_cot, dict)
        else ""
    )
    candidate_vr = (
        candidate_cot.get("visual_recognition", {}).get("text", "")
        if isinstance(candidate_cot, dict)
        else ""
    )
    candidate_kr = (
        candidate_cot.get("knowledge_recall", {}).get("text", "")
        if isinstance(candidate_cot, dict)
        else ""
    )

    if reference_vr:
        lines.append(f"Reference Visual Recognition: {reference_vr}")
    if reference_kr:
        lines.append(f"Reference Knowledge Recall: {reference_kr}")
    if candidate_vr:
        lines.append(f"Candidate Visual Recognition: {candidate_vr}")
    if candidate_kr:
        lines.append(f"Candidate Knowledge Recall: {candidate_kr}")

    return "\n".join(lines).strip() or "No additional support context provided."


def judge_step_hallucination(
    judge_api: CoTAPIWrapper,
    step_key: str,
    question: str,
    ground_truth: str,
    candidate_text: str,
    reference_text: str,
    reference_sample: Dict[str, Any],
    candidate_sample: Dict[str, Any],
    image_dir: Optional[str] = None,
    max_attempts: int = 5,
) -> Dict[str, Any]:
    if step_key not in STEP_PROMPTS:
        raise ValueError(f"Unsupported step key: {step_key}")

    if not candidate_text.strip():
        return {
            "hallucinated": 1,
            "reason": "Candidate step text is empty.",
        }

    support_context = _build_support_context(reference_sample, candidate_sample)
    user_prompt = STEP_PROMPTS[step_key].format(
        question=question,
        ground_truth=ground_truth,
        reference_text=reference_text or "(empty)",
        candidate_text=candidate_text,
        support_context=support_context,
    )

    request_item: Dict[str, Any] = {
        "messages": {
            "system": STEP_HALLUCINATION_JUDGE_SYSTEM_PROMPT,
            "prompt": user_prompt,
        }
    }

    if step_key == "visual_recognition" and image_dir:
        image_paths = resolve_image_paths(candidate_sample, image_dir)
        images = []
        for path in image_paths:
            base64_data = encode_image_to_base64(path)
            if base64_data:
                images.append(
                    {
                        "base64": base64_data,
                        "mime": get_mime_type(path),
                    }
                )
        if images:
            request_item["messages"]["image_base64_list"] = images

    for attempt in range(max_attempts):
        try:
            raw_response = judge_api.generate(request_item)
            parsed = parse_step_hallucination_response(raw_response)
            if parsed is not None:
                return parsed
            logger.warning(
                "Step hallucination judge parse failed for %s (attempt %s/%s), retrying...",
                step_key,
                attempt + 1,
                max_attempts,
            )
        except Exception as error:
            logger.error(
                "Step hallucination judge API call failed for %s (attempt %s/%s): %s",
                step_key,
                attempt + 1,
                max_attempts,
                error,
            )

        if attempt < max_attempts - 1:
            backoff_time = min(2 ** attempt + random.uniform(0, 1), 30)
            time.sleep(backoff_time)

    return dict(DEFAULT_STEP_HALLUCINATION_RESULT)
