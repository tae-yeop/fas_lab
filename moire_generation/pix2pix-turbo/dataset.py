
import os
import json
import torch
import numpy as np
from PIL import Image, ImageFile
import torchvision
from einops import rearrange

def crop_caption(caption, max_words=77):
    sentences = caption.split('. ')
    selected_text = []
    current_word_count = 0

    for sentence in sentences:
        words_in_sentence = len(sentence.split())
        if current_word_count + words_in_sentence <= max_words:
            selected_text.append(sentence)
            current_word_count += words_in_sentence
        else:
            break
    
    final_text = '. '.join(selected_text)
    return final_text


def _list_image_files_recursively(data_dir):
    file_list = []
    for home, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('gt.jpg'):
                file_list.append(os.path.join(home, filename))
    file_list.sort()
    return file_list

def pil_rgb_convert(image):
    if not image.mode == 'RGB':
        image = image.convert('RGB')
    return image

class PairedDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        path='/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/train',
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res=256,
        flip_prob=0.0,
        tokenizer=None
    ):
        super().__init__()
        self.gt_file_paths = _list_image_files_recursively(path)
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.tokenizer = tokenizer
        # caption_path = os.path.join(caption_folder, '.json')
        # with open(caption_path, 'r') as f:
        #     self.captions = json.load(f)
        with open('captions.json', "r") as f:
            self.captions = json.load(f)

    def __len__(self):
        return len(self.gt_file_paths)

    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        clean_path = self.gt_file_paths[idx]
        moire_path = os.path.split(clean_path)[0] + '/' + os.path.split(clean_path)[-1][0:4] + '_moire.jpg'

        filename = os.path.basename(moire_path)
        caption = self.captions[filename]

        image_0 = Image.open(clean_path)
        image_1 = Image.open(moire_path)

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        # image_0 = image_0.resize((self.max_resize_res, self.max_resize_res), Image.Resampling.LANCZOS)
        # image_1 = image_1.resize((self.max_resize_res, self.max_resize_res), Image.Resampling.LANCZOS)
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        
        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)
        # image_0, image_1 = flip(torch.cat((image_0, image_1))).chunk(2)
        # print(caption)

        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {
            "output_pixel_values": image_1,
            "conditioning_pixel_values": image_0,
            "caption": caption,
            "input_ids": input_ids,
        }

if __name__ == '__main__':
    # with open("/purestorage/project/tyk/3_CUProjects/FAS/dreambooth-moire/original/uhdm2/metadata.jsonl", "r") as f:
    #     data = [json.loads(line) for line in f]

    # print(data[:10])

    # updated_data = []
    # with open("/purestorage/project/tyk/3_CUProjects/FAS/pix2pix-turbo/metadata.jsonl", "w") as f:
    #     for item in data:
    #         # print(type(item))
    #         old_text = "Recaptured photo of natural image with TOK pattern noise, featuring "
    #         new_text = "Add moire noise pattern effect to image featuring "
    #         item['prompt'] = item['prompt'].replace(old_text, new_text)
    #         f.write(json.dumps(item) + "\n")

    # caption_path = "/purestorage/project/tyk/3_CUProjects/FAS/pix2pix-turbo/metadata.jsonl"
    # result_dict = {}
    # with open(caption_path, "r") as f:
    #     for line in f:
    #         item_dict = json.loads(line)
    #         result_dict[item_dict['file_name']] = crop_caption(item_dict['prompt'])

    # file_path = 'captions2.json'
    # with open(file_path, 'w') as file:
    #     json.dump(result_dict, file, indent=4)

    # with open('captions.json', "r") as f:
    #     data_loaded = json.load(f)

    # print(type(data_loaded))
    # print(data_loaded['0414_moire.jpg'])

    dataset = PairedDataset()
    next(iter(dataset))
