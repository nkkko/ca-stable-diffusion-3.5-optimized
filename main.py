from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
import torch

model_id = "stabilityai/stable-diffusion-3.5-medium"

# Configure 4-bit quantization
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load the model with quantization
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.float16,
    token="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"
)

# Create pipeline with quantized model
pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.float16,
    token="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"
)

# Enable CPU offloading to further reduce VRAM usage
pipeline.enable_model_cpu_offload()

# Disable safety checker
pipeline.safety_checker = None

# Generate image
image = pipeline(
    "A sophisticated penguin wearing a monocle and top hat, sipping tea at a fancy garden party while other animals look on in disbelief",
    num_inference_steps=40,
    guidance_scale=4.5,
).images[0]
image.save("bruno.png")