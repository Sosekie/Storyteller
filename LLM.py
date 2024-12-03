import torch
from transformers import pipeline
import re

def initialize_llm_model(model_id="meta-llama/Llama-3.2-1B-Instruct"):
    """
    Initialize the LLM model pipeline.
    Returns:
        pipe: The pipeline object for text generation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
    )
    return pipe

def generate_story(pipe, input_sentence, type="simple", order="input-first", max_tokens=128, num_sentences=3):
    """
    Generate a story with a specified number of sentences.
    """
    if type == "simple":
        prompt = f"Write a {num_sentences}-sentence story according to the given sentence. "
    elif type == "detailed":
        prompt = f"You are a storyteller. You need to give a {num_sentences}-sentence story according to the sentence."

    content = f"{input_sentence} {prompt}" if order == "input-first" else f"{prompt} {input_sentence}"
    # print("PROMPT: ", content)

    outputs = pipe(content, max_new_tokens=max_tokens)
    story = outputs[0]["generated_text"]

    # Split into sentences and return the required number
    story = re.split(r'(?<=[.!?])\s+', story)

    # Remove sentences containing unwanted keywords (case insensitive)
    unwanted_keywords = ["sentence", "you need", "prompt", "story"]
    story = [sentence for sentence in story if not any(keyword in sentence.lower() for keyword in unwanted_keywords)]

    # Replace 'he', 'him', 'his', etc. with 'Mickey Mouse'
    story = [re.sub(r'\b(he|him)\b', 'Mickey Mouse', sentence, flags=re.IGNORECASE) for sentence in story]
    story = [re.sub(r'\b(his)\b', "Mickey Mouse's", sentence, flags=re.IGNORECASE) for sentence in story]

    return story[:num_sentences]
