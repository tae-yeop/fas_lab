import torch
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL
from safetensors.torch import load_file
import os


# exp_folder = './moire-SDXL-LoRA-exp1'

repo_id = '/purestorage/project/tyk/3_CUProjects/FAS/dreambooth-moire/original/moire-SDXL-LoRA-exp5-sub1'
pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
).to("cuda")

pipe.load_lora_weights(repo_id, weight_name="pytorch_lora_weights.safetensors")#weight_name=os.path.join(exp_folder,"pytorch_lora_weights.safetensors"))


text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

# embedding_path = hf_hub_download(repo_id=repo_id, filename="embeddings.safetensors", repo_type="model")
emb_path = "/purestorage/project/tyk/3_CUProjects/FAS/dreambooth-moire/original/moire-SDXL-LoRA-exp5-sub1/moire-SDXL-LoRA-exp5-sub1_emb.safetensors"
state_dict = load_file(emb_path)

pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

pipe.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

# instance_token = "<s0><s1>"

# prompt_list = [
#     f"Recaptured photo of natural image with {instance_token} pattern noise, featuring red berries on a tree branch with snow on it",
#     f"Recaptured photo of natural image with {instance_token} pattern noise, featuring a group of sailboats docked at a marina",
#     f"Recaptured photo of natural image with {instance_token} pattern noise, featuring a truck driving down the road",
#     f"Recaptured photo of natural image with {instance_token} pattern noise, featuring a boat in the water",
#     f"Recaptured photo of natural image with {instance_token} pattern noise, featuring a screen shot of a computer screen showing a picture of food",
#     f"Recaptured photo of natural image with {instance_token} pattern noise, featuring the building is made of glass and has a large glass roof",
#     f"Recaptured photo of natural image with {instance_token} pattern noise, featuring a cassette tape with a lighted up face",
#     f"Recaptured photo of natural image with {instance_token} pattern noise, featuring a leaf with yellow and brown stripes on it",
#     f"Recaptured photo of natural image with {instance_token} pattern noise, featuring a bench is sitting in the middle of a park"
# ]

# for idx, prompt in enumerate(prompt_list):
#     image = pipe(prompt=prompt, num_inference_steps=25, cross_attention_kwargs={"scale": 1.0}).images[0]
#     image.save(f'{repo_id}/inference{idx}.png')


print(pipe.scheduler)


pipe.scheduler.set_timesteps(num_inference_steps=44)

print(len(pipe.scheduler.timesteps))


pipe.scheduler.set_timesteps(50)

print(len(pipe.scheduler.timesteps))


from diffusers import UniPCMultistepScheduler, DPMSolverSinglestepScheduler

scheduler = DPMSolverSinglestepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
# print(scheduler)
print(len(scheduler.timesteps))

print(scheduler.config)
scheduler.config.use_karras_sigmas = True
scheduler.set_timesteps(50)
print(len(scheduler.timesteps))
