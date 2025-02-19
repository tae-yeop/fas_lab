# import os
# import pandas as pd
# import glob
# import json
# from PIL import Image


# # out_dir = "./my_folder"
# # os.makedirs(local_dir)


# import pandas as pd
# from pathlib import Path

# # file_paths = ["/home/tyk/output_backup_20.parquet", 
# #               "/home/tyk/output_backup_10.parquet",
# #               "/home/tyk/output_backup_30.parquet"]

# root = "/purestorage/project/tyk/3_CUProjects/FAS/dreambooth-moire/VLM-Captioning-Tools/uhdm_test_caption"
# file_paths = [os.path.join(root, p) for p in os.listdir(root)]

# # print(file_paths)

# dfs = []

# for file_path in file_paths:
#     df = pd.read_parquet(file_path)
#     # 파일 경로 추출
#     file_name = Path(file_path).name
#     # 파일 경로 열 추가
#     df['file_path'] = file_name
#     dfs.append(df)

# merged_df = pd.concat(dfs, ignore_index=True)

# print(merged_df.head())


# def caption_images(input_image):


# caption_prefix = "Recaptured photo of natural image with TOK pattern noise, featuring"
# imgs_and_paths = [(path,Image.open(path)) for path in glob.glob(f"{local_dir}*.jpg")]

# with open(f'{local_dir}metadata.jsonl', 'w') as outfile:
#     for img in imgs_and_paths:
#         caption = caption_prefix + 


import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import glob
import json
from PIL import Image
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b",
                                               cache_dir='/purestorage/project/tyk/3_CUProjects/FAS/i2i-translation/tmp')# .to(device) preprocessor는 device 없음
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",torch_dtype=torch.float16,
cache_dir='/purestorage/project/tyk/3_CUProjects/FAS/i2i-translation/tmp').to(device)


# dataset_folder = '/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/test'
dataset_folder = '/purestorage/project/tyk/3_CUProjects/FAS/dreambooth-moire/original/uhdm2'
caption_prefix = "Recaptured photo of natural image with TOK pattern noise, featuring "
# imgs_and_paths = [(path,Image.open(path)) for path in glob.glob(f"{dataset_folder}*.jpg")]


allowed_extensions = [".jpg", ".png"]
# imgs_and_paths = [(str(path), Image.open(path)) for path in Path(dataset_folder).rglob("*") if path.suffix.lower() in allowed_extensions and path.stem.endswith('moire')]
only_paths = [str(path) for path in Path(dataset_folder).rglob("*") if path.suffix.lower() in allowed_extensions and path.stem.endswith('moire')]
def caption_images(input_image):
    inputs = blip_processor(images=input_image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values
    
    generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

# print(imgs_and_paths[:10])
out_dir = '/purestorage/project/tyk/3_CUProjects/FAS/dreambooth-moire/original/'
# out_dir='/purestorage/project/tyk/3_CUProjects/FAS/dreambooth-moire/VLM-Captioning-Tools/uhdm_test_'

# # imgs_and_paths를 같이 사용 
# with open(f'{out_dir}metadata.jsonl', 'w') as outfile:
#     for img in imgs_and_paths:
#         caption = caption_prefix + caption_images(img[1]).split("\n")[0]
#         entry = {"file_name":img[0].split("/")[-1], "prompt": caption}
#         json.dump(entry, outfile)
#         outfile.write('\n')

# only_paths 사용
with open(f'{out_dir}metadata.jsonl', 'w') as outfile:
    for path in only_paths:
        img = Image.open(path)
        caption = caption_prefix + caption_images(img).split("\n")[0]
        entry = {"file_name":path.split("/")[-1], "prompt": caption}
        json.dump(entry, outfile)
        outfile.write('\n')