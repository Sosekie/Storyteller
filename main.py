from LDM import initialize_ldm_model_quantization, generate_image
from LLM import initialize_llm_model, generate_story

# LLM stroy generation
pipe = initialize_llm_model()
content = "Help me expand this passage into a three-sentence fairy tale, no quotes allowed: Mickey Mouse went to a western restaurant, ordered a roast chicken and ate it. He thought it was very delicious."
# Generate story until it has exactly 3 sentences
while True:
    story = generate_story(pipe, content)
    if len(story) == 3:
        break

# LDM image generation
single_image_pipeline = initialize_ldm_model_quantization()
print("------------STORY BEGIN------------")
for i in range(3):
    print("Paragraph ", i+1, ' : ', story[i])
    with open("prompt.txt", "a") as file:
        file.write(f"Paragraph {i + 1} : {story[i]}\n")
    image = generate_image(single_image_pipeline, story[i][:77])
    image.save(f"Illustration_{i+1}.png")
    print("---------------------------------")
