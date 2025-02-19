from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop

from PIL import Image
from PIL.ImageOps import exif_transpose
from pathlib import Path
import itertools

class PromptDataset(Dataset):
    # prior preservation을 위한 클래스 이미지 제작용
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

    
class DreamBoothDataset(Dataset):
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
        # 'photo of a TOK dog', 'in the style of TOK' 같은건데 만약 TI를 한다면 <si><si+1> 처리가 되서 들어옴
        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.class_prompt = class_prompt
        self.token_abstraction_dict = token_abstraction_dict
        self.train_text_encoder_ti = train_text_encoder_ti

        # 허브에서 얻기
        if dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            dataset = load_dataset(
                dataset_name,
                dataset_config_name,
                cache_dir=cache_dir # 저장할 곳
            )
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            if image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
        # 따로 구축한 데이터셋 
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError("Instance images root doesn't exists.")
            # 미리 이미지 다 오픈
            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]

            # prompt는 None
            # 따로 대응되는 개별적인 prompts없이 그냥 사용하는 듯
            self.custom_instance_prompts = None

        self.instance_images = []
        for img in instance_images:
            # img1, img1, ..., img1, img2, img2, ...
            self.instance_images.extend(itertools.repeat(img, repeats))

        # image processing to prepare for using SD-XL micro-conditioning
        self.original_sizes = []
        self.crop_top_lefts = []
        self.pixel_values = []
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        # 기본적으로 RandomCrop
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # 전처리를 미리 여기서 다 해버리기
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            self.original_sizes.append((image.height, image.width))

            # 전처리 시작
            image = train_resize(image) # 리사이즈
            if args.random_flip and random.random() < 0.5:
                image = train_flip(image)
            if args.center_crop:
            else:
                y,1, x1, h, w = train_crop.get_params(image, (size, size))

            crop_top_left = (y1, x1)
            self.crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            self.pixel_values.append(image)

        # 반복한거 포함한 전체 갯수
        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images
            
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        # 반복되는 것을 고려해서 idx가 사이클돌게 함
        example["instance_images"] = self.pixel_values[index % self.num_instance_images]
        example["original_size"] = self.original_sizes[index % self.num_instance_images]
        example["crop_top_left"] = self.crop_top_lefts[index % self.num_instance_images]

        # 허브에서 받은거에는 promtp가 있다
        if self.custom_instance_prompts:
            caption = self.custom_instance_prompts[index % self.num_instance_images]
            if caption:
                if self.train_text_encoder_ti:
                    # 만약 TI를 한다면 token_abstraction_dict를 전달받았음
                    for token_abs, token_replacement in self.token_abstraction_dict.items():

                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:
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

    # Batchify : 클래스 관련 인풋도 그냥 한 배치로 묶어 버린다 (forawrd 두번 돌지 않게끔)
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

# prompt를 받아서 토큰을 내놓는 유틸리티
def tokenize_prompt(tokenizer, prompt, add_special_tokens=False):
    text_inputs = tokenizer(
        prompt, 
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

# 토큰에서 임베딩과 pooled 임베딩을 얻음
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else: # tokenizer가 없으면
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