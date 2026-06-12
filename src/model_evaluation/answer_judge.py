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

from common import CoTAPIWrapper, check_answer_match
from prompts.judge import (
    ANSWER_JUDGE_SYSTEM_PROMPT,
    ANSWER_JUDGE_USER_PROMPT_TEMPLATE,
    ANSWER_JUDGE_USER_PROMPT_WITH_REPORT_TEMPLATE,
)


logger = logging.getLogger(__name__)

DEFAULT_JUDGE_RESULT = {
    "answer_correct": -1,
    "answer_reasoning": "Judge failed to produce a valid result",
}


def parse_judge_response(raw_response: str) -> Optional[Dict[str, Any]]:
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as error:
        logger.warning("Failed to parse judge JSON: %s", error)
        return None

    if "answer_correct" not in result:
        logger.warning("Missing required field in judge response: answer_correct")
        return None

    reasoning = result.get("answer_reasoning")
    if reasoning is None:
        reasoning = result.get("reasoning", "")

    return {
        "answer_correct": 1 if int(result.get("answer_correct", 0)) == 1 else 0,
        "answer_reasoning": str(reasoning or ""),
    }


def judge_answer_correctness(
    judge_api: CoTAPIWrapper,
    question: str,
    predicted_answer: str,
    ground_truth: str,
    report_text: Optional[str] = None,
    max_attempts: int = 5,
    fallback_to_local: bool = False,
) -> Dict[str, Any]:
    if report_text:
        user_prompt = ANSWER_JUDGE_USER_PROMPT_WITH_REPORT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            predicted_answer=predicted_answer,
            report=report_text,
        )
    else:
        user_prompt = ANSWER_JUDGE_USER_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            predicted_answer=predicted_answer,
        )

    request_item = {
        "messages": {
            "system": ANSWER_JUDGE_SYSTEM_PROMPT,
            "prompt": user_prompt,
        }
    }

    for attempt in range(max_attempts):
        try:
            raw_response = judge_api.generate(request_item)
            parsed = parse_judge_response(raw_response)
            if parsed is not None:
                return parsed
            logger.warning(
                "Judge parse failed (attempt %s/%s), retrying...",
                attempt + 1,
                max_attempts,
            )
        except Exception as error:
            logger.error(
                "Judge API call failed (attempt %s/%s): %s",
                attempt + 1,
                max_attempts,
                error,
            )

        if attempt < max_attempts - 1:
            backoff_time = min(2 ** attempt + random.uniform(0, 1), 30)
            time.sleep(backoff_time)

    if fallback_to_local:
        local_match = check_answer_match(predicted_answer, ground_truth, 0.8)
        return {
            "answer_correct": 1 if local_match else 0,
            "answer_reasoning": "Fallback local string matching after judge failure",
        }

    return dict(DEFAULT_JUDGE_RESULT)
