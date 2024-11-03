import torch
from transformers import pipeline
import re

def initialize_llm_model(model_id="meta-llama/Llama-3.2-1B-Instruct"):
    """
    Initialize the LLM model pipeline.
    
    Returns:
        pipe: The pipeline object for text generation.
    """
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipe

def generate_story(pipe, content):
    """
    Generate a story based on the provided content.
    
    Args:
        pipe: The pipeline object for text generation.
        content: The content to be used for story generation.
    
    Returns:
        str: Generated story content.
    """
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": content},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=256,
    )

    story = outputs[0]["generated_text"][-1]['content']

    story = re.split(r'(?<=[.!])\s+', story)
    story = story[:3]

    return story

# # Example usage:
# pipe = initialize_llm_model()
# content = "Help me expand this passage into a three-sentence fairy tale, no quotes allowed: Mickey Mouse went to a western restaurant, ordered a roast chicken and ate it. He thought it was very delicious."
# story = generate_story(pipe, content)
# # print(story)