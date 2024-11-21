from typing import Dict, List, Tuple, Union, Dict, Any
import pandas as pd
import pyDOE3

from LDM import initialize_ldm_model_quantization, generate_image
from LLM import initialize_llm_model, generate_story

#For LDM choice
SINGLE = True 
PARALLEL = False
# Define parameters and levels
levels: List[int] = [2, 2, 3, 3]  # Number of levels for each parameter
parameters: List[str] = ['prompt_type', 'prompt_order', 'llm_tokens','image_size']
parameter_levels: Dict[str, List[Any]] = {
    'prompt_type': ['simple', 'detailed'],  # 2 levels
    'prompt_order': ['input-first', 'prompt-first'],  # 2 levels
    'llm_tokens': [64,128,256],  # 3 levels
    'image_size':[128,256,512]
}
experiment_matrix = pyDOE3.fullfact(levels)  # Generate the matrix
experiment_configs = pd.DataFrame(
    experiment_matrix,
    columns=parameters
)
experiment_configs.index.name = 'Experiment ID'

REPETITIONS = 1
EXPERIMENT_CONFIGURATIONS = []
input_sentence = "Mickey Mouse went to a western restaurant, ordered a roast chicken, and ate it."
for repetition in range(REPETITIONS):
    for experiment_id, config_row in enumerate(experiment_configs.iterrows()):
        experiment_config = {k: parameter_levels[k][v] for k, v in config_row[1].to_dict().items()}

        EXPERIMENT_CONFIGURATIONS.append(experiment_config)
        print(f"Running experiment: {experiment_id}, repetition: {repetition + 1}")
        print(f"Experiment config: {experiment_config}")

        prompt_type = experiment_config['prompt_type']
        prompt_order = experiment_config['prompt_order']
        llm_tokens = experiment_config['llm_tokens']
        image_size = experiment_config['image_size']


        pipe = initialize_llm_model()

        # Generate story until it has exactly 3 sentences
        while True:
            story = generate_story(pipe, input_sentence, type=prompt_type, order=prompt_order, max_tokens=llm_tokens)
            if len(story) == 3:
                break


        if SINGLE == True:
            # LDM image generation
            single_image_pipeline = initialize_ldm_model_quantization()
            print("------------STORY BEGIN------------")
            for i in range(3):
                print("Paragraph ", i+1, ' : ', story[i])
                with open("rep_{repetition}_{experiment_id}_type_{prompt_type}_order_{prompt_order}_token_{llm_tokens}_size_{image_size}_prompt.txt", "a") as file:
                    file.write(f"Paragraph {i + 1} : {story[i]}\n")
                image = generate_image(single_image_pipeline, story[i][:77],size = image_size)
                image.save(f"rep_{repetition}_{experiment_id}_type_{prompt_type}_order_{prompt_order}_token_{llm_tokens}_size_{image_size}_Illustration_{i+1}.png")
                print("---------------------------------")
        elif PARALLEL == True:
            single_image_pipeline = initialize_ldm_model_quantization()
            print("------------STORY BEGIN------------")
            for i in range(3):
                print("Paragraph ", i+1, ' : ', story[i])
                with open("rep_{repetition}_{experiment_id}_type_{prompt_type}_order_{prompt_order}_token_{llm_tokens}_size_{image_size}_prompt.txt", "a") as file:
                    file.write(f"Paragraph {i + 1} : {story[i]}\n")
            images = generate_image(single_image_pipeline, story,size = image_size)
            for i, image in enumerate(images):
                image.save(f"rep_{repetition}_{experiment_id}_type_{prompt_type}_order_{prompt_order}_token_{llm_tokens}_size_{image_size}_Illustration_{i+1}.png")
                print("---------------------------------")
