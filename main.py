from typing import List, Dict, Any
import pandas as pd
import pyDOE3
import os
import json

from LDM import initialize_ldm_model_quantization, generate_image
from LLM import initialize_llm_model, generate_story

# LDM choice
SINGLE = True
PARALLEL = False

if SINGLE and PARALLEL:
    raise ValueError("SINGLE and PARALLEL modes cannot be enabled simultaneously.")

# Define parameters and levels
levels: List[int] = [2, 2, 3, 3]  # Number of levels for each parameter
parameters: List[str] = ['prompt_type', 'prompt_order', 'llm_tokens', 'image_size']
# parameter_levels: Dict[str, List[Any]] = {
#     'prompt_type': ['simple', 'detailed'],  # 2 levels
#     'prompt_order': ['input-first', 'prompt-first'],  # 2 levels
#     'llm_tokens': [64, 128, 256],  # 3 levels
#     'image_size': [128, 256, 512],  # 3 levels
# }
parameter_levels: Dict[str, List[Any]] = {
    'prompt_type': ['detailed', 'simple'],  # 2 levels, reversed
    'prompt_order': ['prompt-first', 'input-first'],  # 2 levels, reversed
    'llm_tokens': [256, 128, 64],  # 3 levels, reversed
    'image_size': [512, 256, 128],  # 3 levels, reversed
}

experiment_matrix = pyDOE3.fullfact(levels)  # Generate the matrix
if experiment_matrix.shape[1] != len(parameters):
    raise ValueError("Mismatch between the number of parameters and the experiment matrix dimensions.")

experiment_configs = pd.DataFrame(experiment_matrix, columns=parameters)
experiment_configs.index.name = 'Experiment ID'

REPETITIONS = 1
EXPERIMENT_CONFIGURATIONS = []
input_sentence = "Mickey Mouse went to a western restaurant, ordered a roast chicken, and ate it."

output_dir = "output"  # Base directory for output
os.makedirs(output_dir, exist_ok=True)

for repetition in range(REPETITIONS):
    for experiment_id, config_row in enumerate(experiment_configs.iterrows()):
        experiment_config = {k: parameter_levels[k][int(v)] for k, v in config_row[1].to_dict().items()}
        EXPERIMENT_CONFIGURATIONS.append(experiment_config)

        print(f"Running experiment: {experiment_id}, repetition: {repetition + 1}")
        print(f"Experiment config: {experiment_config}")

        # Generate folder name based on experiment configuration
        folder_name = f"promptType-{experiment_config['prompt_type']}_promptOrder-{experiment_config['prompt_order']}_tokens-{experiment_config['llm_tokens']}_imageSize-{experiment_config['image_size']}"
        setting_folder = os.path.join(output_dir, folder_name)
        os.makedirs(setting_folder, exist_ok=True)

        pipe = initialize_llm_model()

        # Generate story with exactly 3 sentences
        max_attempts = 10
        attempts = 0
        story = []
        while attempts < max_attempts:
            story = generate_story(pipe, input_sentence, type=experiment_config['prompt_type'], order=experiment_config['prompt_order'], max_tokens=experiment_config['llm_tokens'], num_sentences=3)
            if len(story) == 3:
                break
            attempts += 1

        if attempts == max_attempts:
            print("Failed to generate a 3-sentence story within the maximum attempts.")
            continue

        if SINGLE:
            single_image_pipeline = initialize_ldm_model_quantization()
            print("------------STORY BEGIN------------")

            # Save the experiment configuration as a JSON file
            config_file_path = os.path.join(setting_folder, "experiment_config.json")
            with open(config_file_path, "w") as config_file:
                json.dump(experiment_config, config_file, indent=4)

            # Save the story into a single text file
            story_file_path = os.path.join(setting_folder, "story.txt")
            with open(story_file_path, "w") as story_file:
                for i in range(3):
                    print(f"Paragraph {i + 1}: {story[i]}")
                    story_file.write(f"Paragraph {i + 1}: {story[i]}\n")

            # Save the input prompt as a text file
            prompt_file_path = os.path.join(setting_folder, "prompt.txt")
            with open(prompt_file_path, "w") as prompt_file:
                prompt_file.write(f"Input Prompt: {input_sentence}\n")

            # Generate images and save them in the setting folder
            images = generate_image(single_image_pipeline, prompt=story, size=experiment_config['image_size'])
            for i, image in enumerate(images):
                image_path = os.path.join(setting_folder, f"Illustration_{i + 1}.png")
                image.save(image_path)
                print(f"Saved image to {image_path}")
                print("---------------------------------")
