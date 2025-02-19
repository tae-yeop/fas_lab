import torch
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file
from diffusers.utils import load_image, make_image_grid
import os



repo_id = '/purestorage/project/tyk/3_CUProjects/FAS/dreambooth-moire/original/moire-SDXL-LoRA-exp5-sub1'
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
).to("cuda")

pipe.load_lora_weights(repo_id, weight_name="pytorch_lora_weights.safetensors")

text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

emb_path = "/purestorage/project/tyk/3_CUProjects/FAS/dreambooth-moire/original/moire-SDXL-LoRA-exp5-sub1/moire-SDXL-LoRA-exp5-sub1_emb.safetensors"
state_dict = load_file(emb_path)

pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

pipe.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

instance_token = "<s0><s1>"

# init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
# prompt = "Recaptured photo of natural image with {instance_token} pattern noise, featuring cat"

# init_image = load_image("/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/test/0479_gt.jpg")
init_image = load_image("https://cdn.newstof.com/news/photo/202306/20936_21680_5514.jpg")
prompt = "Recaptured photo of natural image with a lot of heavily {instance_token} pattern noise featureing a tall tower, which appears to be an observation or communication tower, set against a backdrop of a cloudy sky. The tower is predominantly white, with a red and blue spire at its pinnacle. It is surrounded by lush green trees and foliage. In the foreground, there's a paved pathway with road signs with the contrast of the man-made tower and nature adding a touch of modernity to the natural surroundings."

image = pipe(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
image.save('test8.jpg')