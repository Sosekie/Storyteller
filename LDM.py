from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
import torch

def initialize_ldm_model_original(model_id="stabilityai/stable-diffusion-3.5-large"):
    """
    Initializes the original (non-quantized) version of the Stable Diffusion 3.5 model pipeline.
    Loads the model with bfloat16 precision on the GPU (CUDA).
    """
    # Load the original Stable Diffusion 3 pipeline with bfloat16 precision
    pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipeline = pipeline.to("cuda")  # Move model to GPU for faster inference
    return pipeline

def initialize_ldm_model_quantization(model_id="stabilityai/stable-diffusion-3.5-large"):
    # Configure quantization settings
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load the quantized transformer model
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )
    
    # Load and configure the pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id, 
        transformer=model_nf4,
        torch_dtype=torch.bfloat16
    )
    pipeline.enable_model_cpu_offload()  # Enable CPU offloading to reduce memory usage
    
    return pipeline

# Define a function to generate an image
def generate_image(pipeline, prompt, size, num_inference_steps=28, guidance_scale=4.5, max_sequence_length=512):
    image = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
        height=size,
        width=size,
    ).images[0]
    return image

def generate_image3(pipeline, prompt, size, num_inference_steps=28, guidance_scale=4.5, max_sequence_length=512):
    images = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
        height=size,
        width=size,
        in_channels = 3,
        out_channels = 3,
    ).images
    return images

#https://huggingface.co/docs/diffusers/api/models/sd3_transformer2d#diffusers.SD3Transformer2DModel.in_channels
# # Initialize the LDM model and get the single_image_pipeline
# single_image_pipeline = initialize_ldm_model_quantization()

# paragraph = "Once upon a time, Mickey Mouse wandered into a charming western restaurant nestled by the riverbank. The cozy aroma of spices filled the air, and as he took his seat, the chef recommended a special dish: roast chicken, golden and crispy. Excited by the mouth-watering smell, Mickey eagerly placed his order, feeling his tiny heart race with anticipation."
# image = generate_image(single_image_pipeline, paragraph)
# image.save("paragraph_2.png")
