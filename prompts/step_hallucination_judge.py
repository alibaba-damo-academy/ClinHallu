"""
Unified prompts for step-level hallucination judging.
"""

STEP_HALLUCINATION_JUDGE_SYSTEM_PROMPT = (
    "You are an expert medical AI evaluator. Your task is to determine whether "
    "a single reasoning step in a medical visual question answering chain-of-thought "
    "contains hallucinated, unsupported, or medically incorrect content.\n\n"
    "Judge the candidate step against the provided question, answer, reference step, "
    "and any support context. When images are provided, use them as evidence.\n\n"
    "You MUST make a binary decision and respond with a valid JSON object "
    "(no markdown, no extra text)."
)

STEP_HALLUCINATION_JUDGE_JSON_FORMAT = (
    "{{\n"
    '  "hallucinated": 0 or 1,\n'
    '  "reason": "brief explanation focused on factual support and hallucination risk"\n'
    "}}"
)

VR_HALLUCINATION_JUDGE_USER_PROMPT_TEMPLATE = (
    "Step Type: Visual Recognition\n"
    "Question: {question}\n"
    "Ground Truth Answer: {ground_truth}\n\n"
    "Reference Visual Recognition:\n"
    "{reference_text}\n\n"
    "Candidate Visual Recognition:\n"
    "{candidate_text}\n\n"
    "Determine whether the candidate visual-recognition step contains hallucinated "
    "visual facts, unsupported findings, or materially incorrect observations.\n"
    "Use the reference step as guidance, but prioritize the actual image evidence when available.\n\n"
    "Respond with this exact JSON format:\n"
    + STEP_HALLUCINATION_JUDGE_JSON_FORMAT
)

KR_HALLUCINATION_JUDGE_USER_PROMPT_TEMPLATE = (
    "Step Type: Knowledge Recall\n"
    "Question: {question}\n"
    "Ground Truth Answer: {ground_truth}\n\n"
    "Reference Knowledge Recall:\n"
    "{reference_text}\n\n"
    "Candidate Knowledge Recall:\n"
    "{candidate_text}\n\n"
    "Available support context:\n"
    "{support_context}\n\n"
    "Determine whether the candidate knowledge-recall step introduces hallucinated, "
    "unsupported, irrelevant, or medically incorrect knowledge.\n"
    "Be strict on factual correctness and relevance to the case.\n"
    "Set hallucinated to 1 if there is any material hallucination or unsupported medical claim; otherwise 0.\n\n"
    "Respond with this exact JSON format:\n"
    + STEP_HALLUCINATION_JUDGE_JSON_FORMAT
)

REASONING_HALLUCINATION_JUDGE_USER_PROMPT_TEMPLATE = (
    "Step Type: Reasoning\n"
    "Question: {question}\n"
    "Ground Truth Answer: {ground_truth}\n\n"
    "Reference Reasoning:\n"
    "{reference_text}\n\n"
    "Candidate Reasoning:\n"
    "{candidate_text}\n\n"
    "Available support context:\n"
    "{support_context}\n\n"
    "Determine whether the candidate reasoning step contains hallucinated claims, "
    "unsupported logical jumps, or medically incorrect inference.\n"
    "Focus on whether the conclusion is justified by the available evidence and context.\n"
    "Set hallucinated to 1 if there is any material hallucination or unsupported inference; otherwise 0.\n\n"
    "Respond with this exact JSON format:\n"
    + STEP_HALLUCINATION_JUDGE_JSON_FORMAT
)
