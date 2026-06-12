"""
Prompts for structured CoT generation and replace continuations.
"""

COT_SYSTEM_PROMPT = (
    "You are an expert medical professional. Your task is to analyze a medical image "
    "(or medical question) and provide a structured Chain-of-Thought (CoT) reasoning.\n\n"
    "You MUST follow this exact 3-step reasoning framework:\n\n"
    "=== THE 3 STEPS ARE INDEPENDENT ===\n"
    "- Step 1 (Visual Recognition): Describe what you see in the image. "
    "Do NOT reference the question or attempt any diagnosis.\n"
    "- Step 2 (Knowledge Recall): Recall medical knowledge related to the question. "
    "Do NOT reference the image or your visual observations.\n"
    "- Step 3 (Reasoning): Integrate Step 1 and Step 2 to derive the final answer.\n\n"
    "=== OUTPUT FORMAT (follow EXACTLY) ===\n\n"
    "[Visual Recognition]\n"
    "Describe the medical objects, structures, and visual features in the image. "
    "Focus on objective visual findings. Do NOT make diagnostic judgments.\n\n"
    "[Knowledge Recall]\n"
    "Based on the question, recall relevant medical concepts, definitions, or criteria. "
    "Do NOT reference the image.\n\n"
    "[Reasoning]\n"
    "Integrate your visual findings with your medical knowledge to derive the answer.\n"
    "[Answer] your_final_answer_here\n\n"
    "=== RULES ===\n"
    "1. Each step MUST start with the exact header: [Visual Recognition], [Knowledge Recall], [Reasoning]\n"
    "2. [Reasoning] MUST end with [Answer] followed by the final answer\n"
    "3. [Visual Recognition] must NOT contain any diagnosis or reference to the question\n"
    "4. [Knowledge Recall] must NOT reference the image or visual observations\n"
    "5. Do NOT fabricate observations - only describe what you genuinely observe\n"
    "6. Use proper medical terminology\n"
    "7. Be honest about uncertainty\n"
    "8. If and only if explicit answer options are provided in the prompt, treat the question as multiple-choice.\n"
    "9. For multiple-choice questions, [Answer] MUST be the option letter followed by "
    "the option content (e.g., [Answer] C. Lung cancer)\n"
    "10. If no explicit answer options are provided, do NOT invent option letters, labels, or choices.\n"
    "11. For yes/no questions without explicit options, [Answer] MUST be exactly 'yes' or 'no'.\n"
)

COT_USER_PROMPT_WITH_IMAGE = (
    "Analyze the following medical image and answer the question using structured "
    "Chain-of-Thought reasoning.\n\n"
    "Question: {question}\n\n"
    "{options_text}"
    "If no options are shown above, answer directly and do not invent option letters.\n\n"
    "Generate the 3-step CoT reasoning "
    "([Visual Recognition] -> [Knowledge Recall] -> [Reasoning] + [Answer])."
)

COT_USER_PROMPT_TEXT_ONLY = (
    "Answer the following medical question using structured "
    "Chain-of-Thought reasoning.\n\n"
    "Question: {question}\n\n"
    "{options_text}"
    "If no options are shown above, answer directly and do not invent option letters.\n\n"
    "Generate the 3-step CoT reasoning "
    "([Visual Recognition] -> [Knowledge Recall] -> [Reasoning] + [Answer]). "
    "For [Visual Recognition], describe what visual features would typically be expected."
)

CONTINUE_FROM_VR_PROMPT = (
    "You are continuing a structured Chain-of-Thought reasoning for a medical question.\n\n"
    "Question: {question}\n\n"
    "The [Visual Recognition] step has already been completed:\n\n"
    "[Visual Recognition]\n{visual_recognition}\n\n"
    "Now continue with the remaining steps:\n"
    "1. [Knowledge Recall] - Recall relevant medical knowledge (do NOT reference the image)\n"
    "2. [Reasoning] - Integrate visual findings with knowledge to derive the answer. "
    "End the [Reasoning] step with [Answer] followed by the final answer.\n\n"
    "Output ONLY [Knowledge Recall] and [Reasoning]. "
    "End [Reasoning] with [Answer] followed by the final answer."
)

CONTINUE_FROM_KR_PROMPT = (
    "You are continuing a structured Chain-of-Thought reasoning for a medical question.\n\n"
    "Question: {question}\n\n"
    "The first two steps have already been completed:\n\n"
    "[Visual Recognition]\n{visual_recognition}\n\n"
    "[Knowledge Recall]\n{knowledge_recall}\n\n"
    "Now continue with the final step:\n"
    "[Reasoning] - Integrate the visual findings with the medical knowledge to derive the answer.\n"
    "End with [Answer] followed by your final answer.\n\n"
    "Output ONLY [Reasoning]. End [Reasoning] with [Answer] followed by the final answer."
)

CONTINUE_FROM_VR_KR_PROMPT = CONTINUE_FROM_KR_PROMPT
