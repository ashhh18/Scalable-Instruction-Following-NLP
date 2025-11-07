import time
import os
import sys
import random
import google.generativeai as genai

def load_model():
    genai.configure(api_key="")
    return genai.GenerativeModel("gemini-1.5-flash")

model = load_model()

def get_feedback(prompt):
    response = model.generate_content(prompt)
    return response.text.strip()

def get_grading(prompt):
    response = model.generate_content(prompt)
    return response.text.strip()

def gemini_eval(story):
    prompt = (
        f"In the following exercise, the student is given a beginning of a story. The student needs to complete it into a full story. "
        f"The exercise tests the student's language abilities and creativity. The symbol *** marks the separator between the prescribed beginning and the student's completion. "
        f"Focus on the language abilities and creativity and evaluate the student's completion of the story (following the *** separator), "
        f"Give a concise, comprehensive evaluation on their writing, without discussing the story's content in at most 1 paragraph. "
        f"Do not generate a sample completion or provide feedback at this time. It is ok if the story is incomplete. "
        f"Evaluate the story written so far.\n\n{story}"
    )
    feedback = get_feedback(prompt)

    grading_prompt = (
        f"{story}\n\ncomprehensive Evaluation:{feedback}\n\n"
        f"Now, grade the student's completion in terms of grammar, creativity, consistency with the story's beginning, whether the plot makes sense, and vocabulary diversity.\n\n"
        f"Please provide the grading in JSON format with the keys 'grammar', 'creativity', 'consistency', 'plot_sense', and 'vocabulary_diversity', "
        f"as a number from 1 to 10. DO NOT OUTPUT ANYTHING BUT THE JSON STRICTLY, DIRECTLY START WITH THE JSON BRACKETS."
    )
    return get_grading(grading_prompt)

def gemini_eval_instruct(story):
    prompt = (
        f"In the following exercise, the student is given the details that the story they have to write is supposed to have. "
        f"The exercise tests the student's language abilities and majorly their ability to follow instructions. "
        f"The student's story starts after Story:. Evaluate the student's story, focusing on their language abilities and particularly "
        f"whether they follow the instructions provided and align the story with them. Provide a concise, comprehensive evaluation on their writing, "
        f"without discussing the story's content in at most 1 paragraph. Do not generate a sample completion or provide feedback at this time. "
        f"It is ok if the story is incomplete. Evaluate the story written so far.\n\n{story}"
    )
    feedback = get_feedback(prompt)

    grading_prompt = (
        f"{story}\n\ncomprehensive Evaluation:{feedback}\n\n"
        f"Now, grade the student's story in terms of instruction following ability, grammar, consistency, creativity, and plot sense with a single number between 1 and 10 for each.\n\n"
        f"DO NOT OUTPUT ANYTHING BUT THE JSON STRICTLY, DIRECTLY START WITH THE JSON BRACKETS. The keys would be "
        f"'grammar', 'consistency', 'creativity', 'plot_sense', and 'instruction_ability'."
    )
    return get_grading(grading_prompt)

def gemini_prompt(story, prompt_type):
    prompt_type_map = {0: "factual knowledge", 1: "reasoning ability", 2: "context-tracking"}
    focus = prompt_type_map.get(prompt_type, "unknown")
    
    prompt = (
        f"In the following exercise, the student is given a prompt. The student needs to complete it. "
        f"The exercise solely tests the student's {focus}. The symbol *** marks the separator between the prescribed beginning and the student's completion. "
        f"Evaluate the student's completion of the prompt (following the *** separator), focusing on their {focus}. "
        f"ONLY EVALUATE THE COMPLETION UNTIL THE FIRST END OF SENTENCE IS ENCOUNTERED AFTER ***. IGNORE THE NEXT PARTS OF THE COMPLETION. "
        f"Provide a concise, comprehensive evaluation on their completion, without discussing the prompt's content in at most 1 paragraph.\n\n{story}"
    )
    feedback = get_feedback(prompt)

    grading_prompt = (
        f"{story}\n\ncomprehensive Evaluation:{feedback}\n\n"
        f"Now, grade the student's completion solely in terms of {focus} based on the prompt.\n\n"
        f"Please provide the grading in JSON format with the key '{focus}' as an integer value between 1 and 10. "
        f"DO NOT OUTPUT ANYTHING BUT THE JSON STRICTLY, DIRECTLY START WITH THE JSON BRACKETS."
    )
    return get_grading(grading_prompt)
