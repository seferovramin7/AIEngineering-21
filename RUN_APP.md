# How to Run the Image Generation App

This file explains how to run the `generate_image.py` script to generate images from text prompts.

## 1. Installation

First, you need to install the required Python libraries. Open your terminal and run the following command:

```bash
pip install diffusers transformers torch accelerate
```

## 2. Running the Script

Once the installation is complete, you can use the `generate_image.py` script.

### Basic Usage

The only required argument is `--prompt`.

```bash
python generate_image.py --prompt "A distant futuristic city with tall buildings, flying vehicles, and a soft, glowing ambiance."
```

This will generate an image named `generated_image.png` in the current directory.

### Advanced Usage

You can also specify a negative prompt, an output path, and other parameters:

```bash
python generate_image.py \
    --prompt "A hyper-realistic, detailed photograph of a vintage typewriter on a wooden desk, with a cup of coffee and a stack of old books next to it. The lighting is warm and soft." \
    --negative_prompt "modern, plastic, ugly, poorly lit, blurry, out of focus" \
    --output_path "my_typewriter.png" \
    --num_inference_steps 75 \
    --guidance_scale 8.0
```

This will create an image named `my_typewriter.png` with more specific characteristics.

```