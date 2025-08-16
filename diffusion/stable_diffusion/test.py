import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch

ALLOW_CUDA = False
ALLOW_MPS = True

if torch.cuda.is_available() and ALLOW_CUDA:
    device = "cuda"
elif torch.backends.mps.is_available() and ALLOW_MPS:
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/tokenizer_merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, device)

# IMAGE to IMAGE

input_image = None
image_path = "../data/dog.jpg"
strength = 0.9

# TEXT TO IMAGE

prompt = "A street lamp"
uncond_prompt = "A table lamp"
do_cfg = True
cfg_scale = 7

sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=device,
    idle_device="cpu",
    tokenizer=tokenizer
)

Image.fromarray(output_image).save("output_image.png")
