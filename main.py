from typing import List, Dict, Any
import pandas as pd
import pyDOE3
import os
import json
import time
from LDM import initialize_ldm_model_quantization, generate_image
from LLM import initialize_llm_model, generate_story
from metrics import compute_llm_metrics, compute_story_context_fid, compute_image_metrics
import numpy as np

# LDM choice
SINGLE = True
PARALLEL = False

if SINGLE and PARALLEL:
    raise ValueError("SINGLE and PARALLEL modes cannot be enabled simultaneously.")

# Define parameters and levels
levels: List[int] = [2, 2, 3, 3, 2]  # Number of levels for each parameter
parameters: List[str] = ['prompt_type', 'prompt_order', 'llm_tokens', 'image_size', 'gene_type']
parameter_levels: Dict[str, List[Any]] = {
    'prompt_type': ['simple', 'detailed'],  # 2 levels
    'prompt_order': ['input-first', 'prompt-first'],  # 2 levels
    'llm_tokens': [64, 128, 256],  # 3 levels
    'image_size': [128, 256, 512],  # 3 levels
    'gene_type': ['parallel', 'sequential'],  # 2 levels
}

experiment_matrix = pyDOE3.fullfact(levels)  # Generate the matrix
if experiment_matrix.shape[1] != len(parameters):
    raise ValueError("Mismatch between the number of parameters and the experiment matrix dimensions.")

experiment_configs = pd.DataFrame(experiment_matrix, columns=parameters)
experiment_configs.index.name = 'Experiment ID'

REPETITIONS = 3
EXPERIMENT_CONFIGURATIONS = []
input_sentence = "Mickey Mouse went to a western restaurant, ordered a roast chicken, and ate it."

output_dir = "output"  # Base directory for output
os.makedirs(output_dir, exist_ok=True)

def convert_to_serializable(data):
    """
    Recursively convert numpy data types to Python native types for JSON serialization.
    """
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    else:
        return data

for repetition in range(REPETITIONS):
    start_pipeline_time = time.time()  # Start timing for the entire pipeline

    for experiment_id, config_row in enumerate(experiment_configs.iterrows()):
        experiment_config = {k: parameter_levels[k][int(v)] for k, v in config_row[1].to_dict().items()}
        EXPERIMENT_CONFIGURATIONS.append(experiment_config)

        print(f"Running experiment: {experiment_id}, repetition: {repetition + 1}")
        print(f"Experiment config: {experiment_config}")

        # Generate folder name based on experiment configuration, including repetition
        folder_name = (f"promptType-{experiment_config['prompt_type']}_"
                       f"promptOrder-{experiment_config['prompt_order']}_"
                       f"tokens-{experiment_config['llm_tokens']}_"
                       f"imageSize-{experiment_config['image_size']}_"
                       f"geneType-{experiment_config['gene_type']}_"
                       f"repetition-{repetition + 1}")
        setting_folder = os.path.join(output_dir, folder_name)

        # Check if the folder already exists
        if os.path.exists(setting_folder):
            print(f"Experiment already completed. Skipping: {folder_name}")
            continue

        # Create the folder for this experiment
        os.makedirs(setting_folder, exist_ok=True)

        pipe = initialize_llm_model()

        # Generate story with exactly 3 sentences
        start_llm_time = time.time()
        max_attempts = 10
        attempts = 0
        story = []
        while attempts < max_attempts:
            story = generate_story(pipe, input_sentence, type=experiment_config['prompt_type'], order=experiment_config['prompt_order'], max_tokens=experiment_config['llm_tokens'], num_sentences=3)
            if len(story) == 3:
                break
            attempts += 1
        llm_time = time.time() - start_llm_time  # Time taken for LLM

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

            # Generate images based on gene_type0
            start_ldm_time = time.time()
            if experiment_config['gene_type'] == 'parallel':
                # Generate all images in parallel
                images = generate_image(single_image_pipeline, prompt=story, size=experiment_config['image_size'])
                for i, image in enumerate(images):
                    image_path = os.path.join(setting_folder, f"Illustration_parallel_{i + 1}.png")
                    image.save(image_path)
                    print(f"Saved parallel image to {image_path}")
            elif experiment_config['gene_type'] == 'sequential':
                # Generate images sequentially, one for each sentence
                images = []
                for i, sentence in enumerate(story):
                    image = generate_image(single_image_pipeline, prompt=sentence, size=experiment_config['image_size'])
                    image_path = os.path.join(setting_folder, f"Illustration_sequential_{i + 1}.png")
                    image.save(image_path)
                    images.append(image)
                    print(f"Saved sequential image to {image_path}")
            ldm_time = time.time() - start_ldm_time  # Time taken for LDM

            # Compute metrics
            llm_metrics = compute_llm_metrics(story)
            story_context_fid = compute_story_context_fid(story)
            image_metrics = compute_image_metrics(images)

            # Ensure all values in result_data are JSON serializable
            result_data = {
                "llm_time": float(llm_time),
                "ldm_time": float(ldm_time),
                "pipeline_time": float(time.time() - start_pipeline_time),
                "llm_metrics": llm_metrics,
                "story_context_fid": story_context_fid,
                "image_metrics": image_metrics,
            }

            # Convert to serializable format
            result_data = convert_to_serializable(result_data)

            # Save results to result.json
            result_file_path = os.path.join(setting_folder, "result.json")
            with open(result_file_path, "w") as result_file:
                json.dump(result_data, result_file, indent=4)

            print(f"Results saved to {result_file_path}")
            print("---------------------------------")
