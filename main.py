from diffusers import BitsAndBytesConfig, SD3Transformer2DModel, StableDiffusion3Pipeline
import torch
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get variables from .env
model_id = os.getenv('MODEL_ID')
hf_token = os.getenv('HF_TOKEN')

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
    token=hf_token
)

# Create pipeline with quantized model
pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.float16,
    token=hf_token
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
image.save("image.png")