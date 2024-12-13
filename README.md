# CA Stable Diffusion 3.5 Optimized Demo

This repository contains an optimized implementation of Stable Diffusion 3.5 using 4-bit quantization and CPU offloading for reduced VRAM usage.

## Features

- 4-bit quantization using BitsAndBytes
- CPU offloading for reduced VRAM consumption
- Safety checker disabled for faster inference
- Optimized for memory-constrained systems

## Usage

```bash
python main.py
```

## Configuration

- `num_inference_steps`: Number of denoising steps (default: 40)
- `guidance_scale`: Controls how closely the image follows the prompt (default: 4.5)
- `torch_dtype`: Set to float16 for reduced memory usage

## Memory Optimization

This implementation uses several techniques to reduce VRAM usage:
- 4-bit quantization with NF4 format
- CPU offloading for model components
- Disabled safety checker
- Float16 precision

## Acknowledgments

- Stability AI for the base SD 3.5 model
- Hugging Face for the Diffusers library
- The open-source community

## Contributing

Feel free to submit issues and pull requests to improve the implementation.

## Note

Remember to replace `YOUR_HF_TOKEN` with your actual Hugging Face token in the code.