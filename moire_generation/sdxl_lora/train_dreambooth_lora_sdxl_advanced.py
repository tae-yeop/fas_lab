#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import contextlib
import gc
import hashlib
import itertools
import json
import logging
import math
import os
import random
import re
import shutil
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# imports of the TokenEmbeddingsHandler class
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, hf_hub_download, upload_folder
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

import logging
logger = get_logger(__name__)
#logger.addHandler(logging.FileHandler(logging_path.joinpath('run.log')))

def determine_scheduler_type(pretrained_model_name_or_path, revision):
    model_index_filename = "model_index.json"
    if os.path.isdir(pretrained_model_name_or_path):
        model_index = os.path.join(pretrained_model_name_or_path, model_index_filename)
    else:
        model_index = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=model_index_filename, revision=revision
        )

    with open(model_index, "r") as f:
        scheduler_type = json.load(f)["scheduler"][1]
    return scheduler_type


def save_model_card(
    repo_id: str,
    use_dora: bool,
    images=None,
    base_model=str,
    train_text_encoder=False,
    train_text_encoder_ti=False,
    token_abstraction_dict=None,
    instance_prompt=str,
    validation_prompt=str,
    repo_folder=None,
    vae_path=None,
):
    img_str = "widget:\n"
    lora = "lora" if not use_dora else "dora"
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"""
        - text: '{validation_prompt if validation_prompt else ' ' }'
          output:
            url:
                "image_{i}.png"
        """
    if not images:
        img_str += f"""
        - text: '{instance_prompt}'
        """
    embeddings_filename = f"{repo_folder}_emb"
    instance_prompt_webui = re.sub(r"<s\d+>", "", re.sub(r"<s\d+>", embeddings_filename, instance_prompt, count=1))
    ti_keys = ", ".join(f'"{match}"' for match in re.findall(r"<s\d+>", instance_prompt))
    if instance_prompt_webui != embeddings_filename:
        instance_prompt_sentence = f"For example, `{instance_prompt_webui}`"
    else:
        instance_prompt_sentence = ""
    trigger_str = f"You should use {instance_prompt} to trigger the image generation."
    diffusers_imports_pivotal = ""
    diffusers_example_pivotal = ""
    webui_example_pivotal = ""
    if train_text_encoder_ti:
        trigger_str = (
            "To trigger image generation of trained concept(or concepts) replace each concept identifier "
            "in you prompt with the new inserted tokens:\n"
        )
        diffusers_imports_pivotal = """from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
        """
        diffusers_example_pivotal = f"""embedding_path = hf_hub_download(repo_id='{repo_id}', filename='{embeddings_filename}.safetensors', repo_type="model")
state_dict = load_file(embedding_path)
pipeline.load_textual_inversion(state_dict["clip_l"], token=[{ti_keys}], text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
pipeline.load_textual_inversion(state_dict["clip_g"], token=[{ti_keys}], text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2)
        """
        webui_example_pivotal = f"""- *Embeddings*: download **[`{embeddings_filename}.safetensors` here 💾](/{repo_id}/blob/main/{embeddings_filename}.safetensors)**.
    - Place it on it on your `embeddings` folder
    - Use it by adding `{embeddings_filename}` to your prompt. {instance_prompt_sentence}
    (you need both the LoRA and the embeddings as they were trained together for this LoRA)
    """
        if token_abstraction_dict:
            for key, value in token_abstraction_dict.items():
                tokens = "".join(value)
                trigger_str += f"""
to trigger concept `{key}` → use `{tokens}` in your prompt \n
"""

    yaml = f"""---
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- diffusers-training
- text-to-image
- diffusers
- {lora}
- template:sd-lora
{img_str}
base_model: {base_model}
instance_prompt: {instance_prompt}
license: openrail++
---
"""

    model_card = f"""
# SDXL LoRA DreamBooth - {repo_id}

<Gallery />

## Model description

### These are {repo_id} LoRA adaption weights for {base_model}.

## Download model

### Use it with UIs such as AUTOMATIC1111, Comfy UI, SD.Next, Invoke

- **LoRA**: download **[`{repo_folder}.safetensors` here 💾](/{repo_id}/blob/main/{repo_folder}.safetensors)**.
    - Place it on your `models/Lora` folder.
    - On AUTOMATIC1111, load the LoRA by adding `<lora:{repo_folder}:1>` to your prompt. On ComfyUI just [load it as a regular LoRA](https://comfyanonymous.github.io/ComfyUI_examples/lora/).
{webui_example_pivotal}

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
{diffusers_imports_pivotal}
pipeline = AutoPipelineForText2Image.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')
{diffusers_example_pivotal}
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## Trigger words

{trigger_str}

## Details
All [Files & versions](/{repo_id}/tree/main).

The weights were trained using [🧨 diffusers Advanced Dreambooth Training Script](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py).

LoRA for the text encoder was enabled. {train_text_encoder}.

Pivotal tuning was enabled: {train_text_encoder_ti}.

Special VAE used for training: {vae_path}.

"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand.To load the custom captions, the training set directory needs to follow the structure of a "
            "datasets ImageFolder, containing both the images and the corresponding caption for each image. see: "
            "https://huggingface.co/docs/datasets/image_dataset for more information"
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset. In some cases, a dataset may have more than one configuration (for example "
        "if it contains different subsets of data within, and you only wish to load a specific subset - in that case specify the desired configuration using --dataset_config_name. Leave as "
        "None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A path to local folder containing the training data of instance images. Specify this arg instead of "
        "--dataset_name if you wish to train using a local folder without custom captions. If you wish to train with custom captions please specify "
        "--dataset_name instead.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--token_abstraction",
        type=str,
        default="TOK",
        help="identifier specifying the instance(or instances) as used in instance_prompt, validation prompt, "
        "captions - e.g. TOK. To use multiple identifiers, please specify them in a comma seperated string - e.g. "
        "'TOK,TOK2,TOK3' etc.",
    )

    parser.add_argument(
        "--num_new_tokens_per_abstraction",
        type=int,
        default=2,
        help="number of new tokens inserted to the tokenizers per token_abstraction identifier when "
        "--train_text_encoder_ti = True. By default, each --token_abstraction (e.g. TOK) is mapped to 2 new "
        "tokens - <si><si+1> ",
    )

    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--do_edm_style_training",
        action="store_true",
        help="Flag to conduct training using the EDM formulation as introduced in https://arxiv.org/abs/2206.00364.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--train_text_encoder_ti",
        action="store_true",
        help=("Whether to use textual inversion"),
    )

    parser.add_argument(
        "--train_text_encoder_ti_frac",
        type=float,
        default=0.5,
        help=("The percentage of epochs to perform textual inversion"),
    )

    parser.add_argument(
        "--train_text_encoder_frac",
        type=float,
        default=1.0,
        help=("The percentage of epochs to perform text encoder tuning"),
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=None, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        default=False,
        help=(
            "Wether to train a DoRA as proposed in- DoRA: Weight-Decomposed Low-Rank Adaptation https://arxiv.org/abs/2402.09353. "
            "Note: to use DoRA you need to install peft from main, `pip install git+https://github.com/huggingface/peft.git`"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    if args.train_text_encoder and args.train_text_encoder_ti:
        raise ValueError(
            "Specify only one of `--train_text_encoder` or `--train_text_encoder_ti. "
            "For full LoRA text encoder training check --train_text_encoder, for textual "
            "inversion training check `--train_text_encoder_ti`"
        )

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


# Taken from https://github.com/replicate/cog-sdxl/blob/main/dataset_and_utils.py
class TokenEmbeddingsHandler:
    def __init__(self, text_encoders, tokenizers):
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers

        self.train_ids: Optional[torch.Tensor] = None
        self.inserting_toks: Optional[List[str]] = None
        self.embeddings_settings = {}

    def initialize_new_tokens(self, inserting_toks: List[str]):
        idx = 0
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            assert isinstance(inserting_toks, list), "inserting_toks should be a list of strings."
            assert all(
                isinstance(tok, str) for tok in inserting_toks
            ), "All elements in inserting_toks should be strings."

            self.inserting_toks = inserting_toks
            special_tokens_dict = {"additional_special_tokens": self.inserting_toks}
            tokenizer.add_special_tokens(special_tokens_dict)
            text_encoder.resize_token_embeddings(len(tokenizer))

            self.train_ids = tokenizer.convert_tokens_to_ids(self.inserting_toks)

            # random initialization of new tokens
            std_token_embedding = text_encoder.text_model.embeddings.token_embedding.weight.data.std()

            print(f"{idx} text encodedr's std_token_embedding: {std_token_embedding}")

            text_encoder.text_model.embeddings.token_embedding.weight.data[self.train_ids] = (
                torch.randn(len(self.train_ids), text_encoder.text_model.config.hidden_size)
                .to(device=self.device)
                .to(dtype=self.dtype)
                * std_token_embedding
            )
            self.embeddings_settings[
                f"original_embeddings_{idx}"
            ] = text_encoder.text_model.embeddings.token_embedding.weight.data.clone()
            self.embeddings_settings[f"std_token_embedding_{idx}"] = std_token_embedding

            inu = torch.ones((len(tokenizer),), dtype=torch.bool)
            inu[self.train_ids] = False

            self.embeddings_settings[f"index_no_updates_{idx}"] = inu

            print(self.embeddings_settings[f"index_no_updates_{idx}"].shape)

            idx += 1

    def save_embeddings(self, file_path: str):
        assert self.train_ids is not None, "Initialize new tokens before saving embeddings."
        tensors = {}
        # text_encoder_0 - CLIP ViT-L/14, text_encoder_1 -  CLIP ViT-G/14
        idx_to_text_encoder_name = {0: "clip_l", 1: "clip_g"}
        for idx, text_encoder in enumerate(self.text_encoders):
            assert text_encoder.text_model.embeddings.token_embedding.weight.data.shape[0] == len(
                self.tokenizers[0]
            ), "Tokenizers should be the same."
            new_token_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[self.train_ids]

            # New tokens for each text encoder are saved under "clip_l" (for text_encoder 0), "clip_g" (for
            # text_encoder 1) to keep compatible with the ecosystem.
            # Note: When loading with diffusers, any name can work - simply specify in inference
            tensors[idx_to_text_encoder_name[idx]] = new_token_embeddings
            # tensors[f"text_encoders_{idx}"] = new_token_embeddings

        save_file(tensors, file_path)

    @property
    def dtype(self):
        return self.text_encoders[0].dtype

    @property
    def device(self):
        return self.text_encoders[0].device

    @torch.no_grad()
    def retract_embeddings(self):
        for idx, text_encoder in enumerate(self.text_encoders):
            index_no_updates = self.embeddings_settings[f"index_no_updates_{idx}"]
            text_encoder.text_model.embeddings.token_embedding.weight.data[index_no_updates] = (
                self.embeddings_settings[f"original_embeddings_{idx}"][index_no_updates]
                .to(device=text_encoder.device)
                .to(dtype=text_encoder.dtype)
            )

            # for the parts that were updated, we need to normalize them
            # to have the same std as before
            std_token_embedding = self.embeddings_settings[f"std_token_embedding_{idx}"]

            index_updates = ~index_no_updates
            new_embeddings = text_encoder.text_model.embeddings.token_embedding.weight.data[index_updates]
            off_ratio = std_token_embedding / new_embeddings.std()

            new_embeddings = new_embeddings * (off_ratio**0.1)
            text_encoder.text_model.embeddings.token_embedding.weight.data[index_updates] = new_embeddings


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        dataset_name,
        dataset_config_name,
        cache_dir,
        image_column,
        caption_column,
        train_text_encoder_ti,
        class_data_root=None,
        class_num=None,
        token_abstraction_dict=None,  # token mapping for textual inversion
        size=1024,
        repeats=1,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        self.token_abstraction_dict = token_abstraction_dict
        self.train_text_encoder_ti = train_text_encoder_ti
        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                dataset_name,
                dataset_config_name,
                cache_dir=cache_dir,
            )
            if "test" in dataset_name:
                split = "test"
            else:
                split = "train"
            # Preprocessing the datasets.
            column_names = dataset[split].column_names
            print('column_names', column_names)
            # 6. Get the column names for input/target.
            if image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            instance_images = dataset[split][image_column]

            if caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset[split][caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")

            allowed_extensions = [".jpg", ".png"]
            instance_images = [Image.open(path) for path in self.instance_data_root.rglob("*") if path.suffix.lower() in allowed_extensions and path.stem.endswith('moire')]

            # data_list = [Image.open(p) for p in list(self.instance_data_root.iterdir()) if p.is_file() and p.stem.endswith('moire')]
            # instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]
            self.custom_instance_prompts = None

            # for folder in ['train', 'test']
            
        print('repeatationg')
        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        print('prepare')
        # image processing to prepare for using SD-XL micro-conditioning
        self.original_sizes = []
        self.crop_top_lefts = []
        self.pixel_values = []
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        print('preprecessing entry')
        # 처음에는 바른데 점점 느려진다 : 912번에서 멈춘다 
        # 메모리 늘리니깐 괜찮아 진다 
        for idx, image in enumerate(self.instance_images):
            print(f'idx : {idx}')
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            self.original_sizes.append((image.height, image.width))
            image = train_resize(image)
            print('done resize')
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            self.crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            print('done tranforms')
            self.pixel_values.append(image)

        print('preprecessing done')
        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())

            self.original_sizes_class_imgs = []
            self.crop_top_lefts_class_imgs = []
            self.pixel_values_class_imgs = []
            self.class_images = [Image.open(path) for path in self.class_images_path]
            for image in self.class_images:
                image = exif_transpose(image)
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                self.original_sizes_class_imgs.append((image.height, image.width))
                image = train_resize(image)
                if args.random_flip and random.random() < 0.5:
                    # flip
                    image = train_flip(image)
                if args.center_crop:
                    y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                    x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                    image = train_crop(image)
                else:
                    y1, x1, h, w = train_crop.get_params(image, (args.resolution, args.resolution))
                    image = crop(image, y1, x1, h, w)
                crop_top_left = (y1, x1)
                self.crop_top_lefts_class_imgs.append(crop_top_left)
                image = train_transforms(image)
                self.pixel_values_class_imgs.append(image)

            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        example["instance_images"] = self.pixel_values[index % self.num_instance_images]
        example["original_size"] = self.original_sizes[index % self.num_instance_images]
        example["crop_top_left"] = self.crop_top_lefts[index % self.num_instance_images]

        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                if self.train_text_encoder_ti:
                    # replace instances of --token_abstraction in caption with the new tokens: "<si><si+1>" etc.
                    for token_abs, token_replacement in self.token_abstraction_dict.items():
                        caption = caption.replace(token_abs, "".join(token_replacement))
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # costum prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        if self.class_data_root:
            example["class_prompt"] = self.class_prompt
            example["class_images"] = self.pixel_values_class_imgs[index % self.num_class_images]
            example["class_original_size"] = self.original_sizes_class_imgs[index % self.num_class_images]
            example["class_crop_top_left"] = self.crop_top_lefts_class_imgs[index % self.num_class_images]

        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]
        original_sizes += [example["class_original_size"] for example in examples]
        crop_top_lefts += [example["class_crop_top_left"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, add_special_tokens=False):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    if args.do_edm_style_training and args.snr_gamma is not None:
        raise ValueError("Min-SNR formulation is not supported when conducting EDM-style training.")

    logging_dir = Path(args.output_dir, args.logging_dir)

    # ----------------------------------
    # Distributed Setting
    # ----------------------------------
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    # --------------------------------
    # prior_preservation
    # -------------------------------
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                revision=args.revision,
                variant=args.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        model_id = args.hub_model_id or Path(args.output_dir).name
        repo_id = None
        if args.push_to_hub:
            repo_id = create_repo(repo_id=model_id, exist_ok=True, token=args.hub_token).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        variant=args.variant,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        variant=args.variant,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    scheduler_type = determine_scheduler_type(args.pretrained_model_name_or_path, args.revision)
    if "EDM" in scheduler_type:
        args.do_edm_style_training = True
        noise_scheduler = EDMEulerScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        logger.info("Performing EDM-style training!")
    elif args.do_edm_style_training:
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        logger.info("Performing EDM-style training!")
    else:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    latents_mean = latents_std = None
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
        latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    if args.train_text_encoder_ti:
        # we parse the provided token identifier (or identifiers) into a list. s.t. - "TOK" -> ["TOK"], "TOK,
        # TOK2" -> ["TOK", "TOK2"] etc.
        token_abstraction_list = "".join(args.token_abstraction.split()).split(",")
        logger.info(f"list of token identifiers: {token_abstraction_list}")

        token_abstraction_dict = {}
        token_idx = 0
        for i, token in enumerate(token_abstraction_list):
            token_abstraction_dict[token] = [
                f"<s{token_idx + i + j}>" for j in range(args.num_new_tokens_per_abstraction)
            ]
            token_idx += args.num_new_tokens_per_abstraction - 1

        # replace instances of --token_abstraction in --instance_prompt with the new tokens: "<si><si+1>" etc.
        for token_abs, token_replacement in token_abstraction_dict.items():
            args.instance_prompt = args.instance_prompt.replace(token_abs, "".join(token_replacement))
            if args.with_prior_preservation:
                args.class_prompt = args.class_prompt.replace(token_abs, "".join(token_replacement))
            if args.validation_prompt:
                args.validation_prompt = args.validation_prompt.replace(token_abs, "".join(token_replacement))
                print("validation prompt:", args.validation_prompt)
        # initialize the new tokens for textual inversion
        embedding_handler = TokenEmbeddingsHandler(
            [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two]
        )
        inserting_toks = []
        for new_tok in token_abstraction_dict.values():
            inserting_toks.extend(new_tok)
        embedding_handler.initialize_new_tokens(inserting_toks=inserting_toks)

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)

    # The VAE is always in float32 to avoid NaN losses.
    vae.to(accelerator.device, dtype=torch.float32)

    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        use_dora=args.use_dora,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            use_dora=args.use_dora,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

    # if we use textual inversion, we freeze all parameters except for the token embeddings
    # in text encoder
    elif args.train_text_encoder_ti:
        text_lora_parameters_one = []
        # 토큰 임베딩 레이어만 학습할 수 있게 한다
        # 학습하는 부분은 torch.float32
        for name, param in text_encoder_one.named_parameters():
            if "token_embedding" in name:
                # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
                param.data = param.to(dtype=torch.float32)
                param.requires_grad = True
                text_lora_parameters_one.append(param)
            else:
                param.requires_grad = False
        text_lora_parameters_two = []
        for name, param in text_encoder_two.named_parameters():
            if "token_embedding" in name:
                # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
                param.data = param.to(dtype=torch.float32)
                param.requires_grad = True
                text_lora_parameters_two.append(param)
            else:
                param.requires_grad = False

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        # compile 걸린것도 빼내기
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Lora layer만 저장하기
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    if args.train_text_encoder:
                        text_encoder_one_lora_layers_to_save = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                elif isinstance(model, type(unwrap_model(text_encoder_two))):
                    if args.train_text_encoder:
                        text_encoder_two_lora_layers_to_save = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )
        if args.train_text_encoder_ti:
            embedding_handler.save_embeddings(f"{output_dir}/{args.output_dir}_emb.safetensors")

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)

        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
            cast_training_params(models) # requries_grad True인것만 바꿔주는 듯

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models, dtype=torch.float32)

    unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))

    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
        text_lora_parameters_two = list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))

    # If neither --train_text_encoder nor --train_text_encoder_ti, text_encoders remain frozen during training
    # 만약 encoder 학습, TI 학습 둘다 안하면 freeze
    freeze_text_encoder = not (args.train_text_encoder or args.train_text_encoder_ti)

    # Optimization parameters
    unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": args.learning_rate}
    if not freeze_text_encoder:
        # different learning rate for text encoder and unet
        # 만약 text encoder도 학습하면 learning rate 다르게 설정
        text_lora_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder
            if args.adam_weight_decay_text_encoder
            else args.adam_weight_decay,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        text_lora_parameters_two_with_lr = {
            "params": text_lora_parameters_two,
            "weight_decay": args.adam_weight_decay_text_encoder
            if args.adam_weight_decay_text_encoder
            else args.adam_weight_decay,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            unet_lora_parameters_with_lr,
            text_lora_parameters_one_with_lr,
            text_lora_parameters_two_with_lr,
        ]
    else:
        params_to_optimize = [unet_lora_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        # 만약에 text_encoder에 별도의 learning_rate 적용했다면 prodigy에 맞춰준다
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the unet and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one and text_encoder_parameters_two to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate
            params_to_optimize[2]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # -----------------------
    # Dataset
    # -----------------------
    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        cache_dir=args.cache_dir,
        image_column=args.image_column,
        train_text_encoder_ti=args.train_text_encoder_ti,
        caption_column=args.caption_column,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        token_abstraction_dict=token_abstraction_dict if args.train_text_encoder_ti else None,
        class_num=args.num_class_images,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
    )
    print('train_dataset done')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )
    print('train_dataloader done')
    # Computes additional embeddings/ids required by the SDXL UNet.
    # regular text embeddings (when `train_text_encoder` is not True)
    # pooled text embeddings
    # time ids

    # SDXL에 추가적으로 들어가는 임베딩, ids 계산
    def compute_time_ids(crops_coords_top_left, original_size=None):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        if original_size is None:
            original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if freeze_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )

    # Handle class prompt for prior-preservation.
    if args.with_prior_preservation:
        if freeze_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds = compute_text_embeddings(
                args.class_prompt, text_encoders, tokenizers
            )

    # Clear the memory here
    if freeze_text_encoder and not train_dataset.custom_instance_prompts:
        del tokenizers, text_encoders
        gc.collect()
        torch.cuda.empty_cache()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    # if --train_text_encoder_ti we need add_special_tokens to be True fo textual inversion
    add_special_tokens = True if args.train_text_encoder_ti else False

    if not train_dataset.custom_instance_prompts:
        if freeze_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            unet_add_text_embeds = instance_pooled_prompt_embeds
            if args.with_prior_preservation:
                prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
                unet_add_text_embeds = torch.cat([unet_add_text_embeds, class_pooled_prompt_embeds], dim=0)
        # if we're optmizing the text encoder (both if instance prompt is used for all images or custom prompts) we need to tokenize and encode the
        # batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(tokenizer_one, args.instance_prompt, add_special_tokens)
            tokens_two = tokenize_prompt(tokenizer_two, args.instance_prompt, add_special_tokens)
            if args.with_prior_preservation:
                class_tokens_one = tokenize_prompt(tokenizer_one, args.class_prompt, add_special_tokens)
                class_tokens_two = tokenize_prompt(tokenizer_two, args.class_prompt, add_special_tokens)
                tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)
    # 미리 VAE에 보내서 캐싱해둠
    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=torch.float32
                )
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
        # validation 안할거면 vae 필요 x
        if args.validation_prompt is None:
            del vae
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 스케쥴러
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    # (total_steps/epoch) / (update_num) : 1 epcoh당 step 밟는 총 횟수
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # 최대 step 없으면 epoch을 이용해서 계산함
    if args.max_train_steps is None:
        # total steps 계산
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    print('scheduler done')

    # ------------------------
    # Training setup
    # ------------------------
    # Prepare everything with our `accelerator`.
    # text encoder를 학습하거나 TI 
    if not freeze_text_encoder:
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    # prepare 후 분산처리 때문에 total step 변경될 수 있음
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps: # 한번 total step 변경된적 있으면
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    # 전체 epoch을 total step에 맞춰 다시 계산 => 목표는 total step만큼 학습하는 것
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # Tracker를 통해 config 저장
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth-lora-sd-xl", config=vars(args))

    # Train!
    # 전체 배치 사이즈 (effective batch size)
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    print("***** Running training *****")
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # 체크포인트 처리
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # 특정한 step에서 시작
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        # 가장 최신이라면 소팅해서 얻음
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        # 없는 경우 처리
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        # 있는 경우 path에서 로딩한다, 이름에서 step을 얻는다
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    # scratch에서 시작
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # EDM 스타일 학습시 필요한 sigma 얻기
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        # TODO: revisit other sampling algorithms
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # text enc와 ti의 경우 학습할 epoch 설정
    if args.train_text_encoder:
        num_train_epochs_text_encoder = int(args.train_text_encoder_frac * args.num_train_epochs)
    elif args.train_text_encoder_ti:  # args.train_text_encoder_ti
        num_train_epochs_text_encoder = int(args.train_text_encoder_ti_frac * args.num_train_epochs)
    # flag used for textual inversion
    # --------------------------
    # Training loop
    # --------------------------
    pivoted = False
    for epoch in range(first_epoch, args.num_train_epochs):
        # 텍스트 관련
        # if performing any kind of optimization of text_encoder params
        if args.train_text_encoder or args.train_text_encoder_ti:
            # 텍스트 관련 학습 끝 탈출
            if epoch == num_train_epochs_text_encoder:
                print("PIVOT HALFWAY", epoch)
                # stopping optimization of text_encoder params
                # this flag is used to reset the optimizer to optimize only on unet params
                pivoted = True

            else:
                # still optimizing the text encoder
                text_encoder_one.train()
                text_encoder_two.train()
                # set top parameter requires_grad = True for gradient checkpointing works
                if args.train_text_encoder:
                    text_encoder_one.text_model.embeddings.requires_grad_(True)
                    text_encoder_two.text_model.embeddings.requires_grad_(True)

        unet.train()
        for step, batch in enumerate(train_dataloader):
            if pivoted:
                # stopping optimization of text_encoder params
                # re setting the optimizer to optimize only on unet params
                # 텍스트 관련 완료시 학습 멈추기
                optimizer.param_groups[1]["lr"] = 0.0
                optimizer.param_groups[2]["lr"] = 0.0

            with accelerator.accumulate(unet):
                prompts = batch["prompts"]
                # -------------------------
                # Input : Prompt Token
                # -------------------------
                # encode batch prompts when custom prompts are provided for each image -
                # 만약 텍스트 관련은 학습 안하면 => torch.no_grad로 뽑아냄
                if train_dataset.custom_instance_prompts:
                    if freeze_text_encoder:
                        prompt_embeds, unet_add_text_embeds = compute_text_embeddings(
                            prompts, text_encoders, tokenizers
                        )

                    else:
                        tokens_one = tokenize_prompt(tokenizer_one, prompts, add_special_tokens)
                        tokens_two = tokenize_prompt(tokenizer_two, prompts, add_special_tokens)

                # ---------------------------------
                # Input : Latent
                # ---------------------------------
                if args.cache_latents:
                    model_input = latents_cache[step].sample()
                else:
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                # mean, std가 vae cfg에 없엇던 경우 그냥 scaling_factor로 곱한다
                if latents_mean is None and latents_std is None:
                    model_input = model_input * vae.config.scaling_factor
                    if args.pretrained_vae_model_name_or_path is None:
                        model_input = model_input.to(weight_dtype)
                else:
                    latents_mean = latents_mean.to(device=model_input.device, dtype=model_input.dtype)
                    latents_std = latents_std.to(device=model_input.device, dtype=model_input.dtype)
                    model_input = (model_input - latents_mean) * vae.config.scaling_factor / latents_std
                    model_input = model_input.to(dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (model_input.shape[0], model_input.shape[1], 1, 1), device=model_input.device
                    )

                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                if not args.do_edm_style_training:
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()
                else:
                    # in EDM formulation, the model is conditioned on the pre-conditioned noise levels
                    # instead of discrete timesteps, so here we sample indices to get the noise levels
                    # from `scheduler.timesteps`
                    indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
                    timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                # Noisy latent 만들기
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                # For EDM-style training, we first obtain the sigmas based on the continuous timesteps.
                # We then precondition the final model inputs based on these sigmas instead of the timesteps.
                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # EDM 스타일의 경우 sigma를 이용한다
                if args.do_edm_style_training:
                    sigmas = get_sigmas(timesteps, len(noisy_model_input.shape), noisy_model_input.dtype)
                    if "EDM" in scheduler_type:
                        inp_noisy_latents = noise_scheduler.precondition_inputs(noisy_model_input, sigmas)
                    else:
                        inp_noisy_latents = noisy_model_input / ((sigmas**2 + 1) ** 0.5)

                # ------------------------------------------
                # Input : Time ids
                # ------------------------------------------
                add_time_ids = torch.cat(
                    [
                        compute_time_ids(original_size=s, crops_coords_top_left=c)
                        for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
                    ]
                )

                # instance prompt 쓰는지 안쓰는지 따라 텍스트 임베딩 반복 정도
                # Calculate the elements to repeat depending on the use of prior-preservation and custom captions.
                if not train_dataset.custom_instance_prompts:
                    elems_to_repeat_text_embeds = bsz // 2 if args.with_prior_preservation else bsz

                else:
                    elems_to_repeat_text_embeds = 1
                # -------------------------------------------
                # Forward
                # -------------------------------------------
                # Predict the noise residual
                if freeze_text_encoder:
                    unet_added_conditions = {
                        "time_ids": add_time_ids,
                        # "time_ids": add_time_ids.repeat(elems_to_repeat_time_ids, 1),
                        "text_embeds": unet_add_text_embeds.repeat(elems_to_repeat_text_embeds, 1),
                    }
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(
                        inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                    ).sample
                # 텍스트 관련 학습 한다면 토큰으로 부터 임베딩 얻기
                else:
                    unet_added_conditions = {"time_ids": add_time_ids}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[tokens_one, tokens_two],
                    )
                    unet_added_conditions.update(
                        {"text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)}
                    )
                    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)
                    model_pred = unet(
                        inp_noisy_latents if args.do_edm_style_training else noisy_model_input,
                        timesteps,
                        prompt_embeds_input,
                        added_cond_kwargs=unet_added_conditions,
                    ).sample

                weighting = None
                if args.do_edm_style_training:
                    # Similar to the input preconditioning, the model predictions are also preconditioned
                    # on noised model inputs (before preconditioning) and the sigmas.
                    # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                    if "EDM" in scheduler_type:
                        model_pred = noise_scheduler.precondition_outputs(noisy_model_input, model_pred, sigmas)
                    else:
                        if noise_scheduler.config.prediction_type == "epsilon":
                            model_pred = model_pred * (-sigmas) + noisy_model_input
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            model_pred = model_pred * (-sigmas / (sigmas**2 + 1) ** 0.5) + (
                                noisy_model_input / (sigmas**2 + 1)
                            )
                    # We are not doing weighting here because it tends result in numerical problems.
                    # See: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
                    # There might be other alternatives for weighting as well:
                    # https://github.com/huggingface/diffusers/pull/7126#discussion_r1505404686
                    if "EDM" not in scheduler_type:
                        weighting = (sigmas**-2.0).float()

                # Get the target for loss depending on the prediction type
                # Target
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = model_input if args.do_edm_style_training else noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = (
                        model_input
                        if args.do_edm_style_training
                        else noise_scheduler.get_velocity(model_input, noise, timesteps)
                    )
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Prior preservation
                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    if weighting is not None:
                        prior_loss = torch.mean(
                            (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                                target_prior.shape[0], -1
                            ),
                            1,
                        )
                        prior_loss = prior_loss.mean()
                    else:
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                if args.snr_gamma is None:
                    if weighting is not None:
                        loss = torch.mean(
                            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                                target.shape[0], -1
                            ),
                            1,
                        )
                        loss = loss.mean()
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.

                    if args.with_prior_preservation:
                        # if we're using prior preservation, we calc snr for instance loss only -
                        # and hence only need timesteps corresponding to instance images
                        snr_timesteps, _ = torch.chunk(timesteps, 2, dim=0)
                    else:
                        snr_timesteps = timesteps

                    snr = compute_snr(noise_scheduler, snr_timesteps)
                    base_weight = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(snr_timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective needs to be floored to an SNR weight of one.
                        mse_loss_weights = base_weight + 1
                    else:
                        # Epsilon and sample both use the same loss weights.
                        mse_loss_weights = base_weight

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet_lora_parameters, text_lora_parameters_one, text_lora_parameters_two)
                        if (args.train_text_encoder or args.train_text_encoder_ti)
                        else unet_lora_parameters
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # every step, we reset the embeddings to the original embeddings.
                if args.train_text_encoder_ti:
                    embedding_handler.retract_embeddings()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # create pipeline
                if freeze_text_encoder:
                    text_encoder_one = text_encoder_cls_one.from_pretrained(
                        args.pretrained_model_name_or_path,
                        subfolder="text_encoder",
                        revision=args.revision,
                        variant=args.variant,
                    )
                    text_encoder_two = text_encoder_cls_two.from_pretrained(
                        args.pretrained_model_name_or_path,
                        subfolder="text_encoder_2",
                        revision=args.revision,
                        variant=args.variant,
                    )
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    tokenizer=tokenizer_one,
                    tokenizer_2=tokenizer_two,
                    text_encoder=accelerator.unwrap_model(text_encoder_one),
                    text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                    unet=accelerator.unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )

                # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                scheduler_args = {}

                if not args.do_edm_style_training:
                    if "variance_type" in pipeline.scheduler.config:
                        variance_type = pipeline.scheduler.config.variance_type

                        if variance_type in ["learned", "learned_range"]:
                            variance_type = "fixed_small"

                        scheduler_args["variance_type"] = variance_type

                    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipeline.scheduler.config, **scheduler_args
                    )

                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
                pipeline_args = {"prompt": args.validation_prompt}
                inference_ctx = (
                    contextlib.nullcontext()
                    if "playground" in args.pretrained_model_name_or_path
                    else torch.cuda.amp.autocast()
                )

                with inference_ctx:
                    images = [
                        pipeline(**pipeline_args, generator=generator).images[0]
                        for _ in range(args.num_validation_images)
                    ]

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        if args.train_text_encoder:
            text_encoder_one = accelerator.unwrap_model(text_encoder_one)
            text_encoder_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_one.to(torch.float32))
            )
            text_encoder_two = accelerator.unwrap_model(text_encoder_two)
            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_two.to(torch.float32))
            )
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )

        if args.train_text_encoder_ti:
            embeddings_path = f"{args.output_dir}/{args.output_dir}_emb.safetensors"
            embedding_handler.save_embeddings(embeddings_path)

        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            # Final inference
            # Load previous pipeline
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )

            # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
            scheduler_args = {}

            if not args.do_edm_style_training:
                if "variance_type" in pipeline.scheduler.config:
                    variance_type = pipeline.scheduler.config.variance_type

                    if variance_type in ["learned", "learned_range"]:
                        variance_type = "fixed_small"

                    scheduler_args["variance_type"] = variance_type

                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config, **scheduler_args
                )

            # load attention processors
            pipeline.load_lora_weights(args.output_dir)

            # load new tokens
            if args.train_text_encoder_ti:
                state_dict = load_file(embeddings_path)
                all_new_tokens = []
                for key, value in token_abstraction_dict.items():
                    all_new_tokens.extend(value)
                pipeline.load_textual_inversion(
                    state_dict["clip_l"],
                    token=all_new_tokens,
                    text_encoder=pipeline.text_encoder,
                    tokenizer=pipeline.tokenizer,
                )
                pipeline.load_textual_inversion(
                    state_dict["clip_g"],
                    token=all_new_tokens,
                    text_encoder=pipeline.text_encoder_2,
                    tokenizer=pipeline.tokenizer_2,
                )

            # run inference
            pipeline = pipeline.to(accelerator.device)
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
            images = [
                pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                for _ in range(args.num_validation_images)
            ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

        # Conver to WebUI format
        lora_state_dict = load_file(f"{args.output_dir}/pytorch_lora_weights.safetensors")
        peft_state_dict = convert_all_state_dict_to_peft(lora_state_dict)
        kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
        save_file(kohya_state_dict, f"{args.output_dir}/{args.output_dir}.safetensors")

        save_model_card(
            model_id if not args.push_to_hub else repo_id,
            use_dora=args.use_dora,
            images=images,
            base_model=args.pretrained_model_name_or_path,
            train_text_encoder=args.train_text_encoder,
            train_text_encoder_ti=args.train_text_encoder_ti,
            token_abstraction_dict=train_dataset.token_abstraction_dict,
            instance_prompt=args.instance_prompt,
            validation_prompt=args.validation_prompt,
            repo_folder=args.output_dir,
            vae_path=args.pretrained_vae_model_name_or_path,
        )
        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
