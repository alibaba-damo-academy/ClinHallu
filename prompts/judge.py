"""
Unified prompts for final-answer judging.
"""

ANSWER_JUDGE_SYSTEM_PROMPT = (
    "You are an expert medical AI evaluator. Your task is to determine whether "
    "a model's predicted answer is correct by comparing it against the ground truth answer "
    "for a medical visual question answering task.\n\n"
    "You MUST respond with a valid JSON object (no markdown, no extra text)."
)

ANSWER_JUDGE_USER_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Ground Truth Answer: {ground_truth}\n"
    "Model's Predicted Answer: {predicted_answer}\n\n"
    "Determine whether the predicted answer is semantically correct.\n"
    "Be lenient on wording but strict on medical facts.\n\n"
    "Respond with this exact JSON format:\n"
    "{{\n"
    '  "answer_correct": 0 or 1,\n'
    '  "answer_reasoning": "brief explanation of why the answer is correct or wrong"\n'
    "}}"
)

ANSWER_JUDGE_USER_PROMPT_WITH_REPORT_TEMPLATE = (
    "Question: {question}\n"
    "Ground Truth Answer: {ground_truth}\n"
    "Model's Predicted Answer: {predicted_answer}\n\n"
    "Additional report context:\n"
    "{report}\n\n"
    "Determine whether the predicted answer is semantically correct.\n"
    "Use the report only as supplementary context, not as a replacement for the ground truth.\n\n"
    "Respond with this exact JSON format:\n"
    "{{\n"
    '  "answer_correct": 0 or 1,\n'
    '  "answer_reasoning": "brief explanation of why the answer is correct or wrong"\n'
    "}}"
)
