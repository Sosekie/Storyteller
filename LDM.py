from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
import torch

def initialize_ldm_model_quantization(model_id="stabilityai/stable-diffusion-3.5-large"):
    """
    Initializes a quantized Stable Diffusion 3.5 model pipeline.
    """
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
    )
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=model_nf4,
        torch_dtype=torch.bfloat16,
    )
    pipeline.enable_model_cpu_offload()
    return pipeline

def generate_image(pipeline, prompt, size, num_inference_steps=28, guidance_scale=4.5, max_sequence_length=512):
    """
    Generate one or multiple images using the Stable Diffusion pipeline.

    Args:
        pipeline: The Stable Diffusion pipeline object.
        prompt: A string (single prompt) or a list of strings (multiple prompts).
        size: The height and width of the generated image(s).
        num_inference_steps: Number of inference steps.
        guidance_scale: Guidance scale for classifier-free guidance.
        max_sequence_length: Maximum sequence length for text input.

    Returns:
        List of generated images (or a single image if only one prompt is provided).
    """
    if isinstance(prompt, str):
        prompt = [prompt]  # Convert single prompt to a list

    # Generate images for each prompt in parallel
    results = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
        height=size,
        width=size,
    ).images

    return results if len(results) > 1 else results[0]
