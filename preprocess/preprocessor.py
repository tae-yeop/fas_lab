import yaml
import argparse
import glob
import os

from label_func import LABEL_REGISTRY

def is_failed_images(filename):
    # The flag is for image that did not detect face
    flag = filename.find('_f.png')
    if (flag == -1):
        return False
    else:
        return True

def get_images(folder_path, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, f"**/*{ext}"), recursive=True))
    return image_files

def process_dataset(
    dataset_name: str,
    src_dirs: List[str],
    dest_dir: str,
    label_function: Callable,
    label_func_name: str,
    use_keypoint: bool = True
):

    filename_list = []

    for src in src_dirs:
        if not os.path.exists(src):
            raise OSError(2, 'No such file or directory', src)
        
        filename_list.extend(get_images(src))

    if not filename_list:
        raise ValueError(f"입력 폴더({src_dirs})에 이미지가 없습니다.")

    # 저장할 폴더 생성
    output_folder = os.path.join(dest_dir, dataset_name)
    os.makedirs(output_folder, exist_ok=True)

    # json 생성
    f_all = open(os.path.join(destination, 'labels.json'), 'w')
    all_final_json = []

    for filename in filename_list:
        if is_failed_images(filename=filename):
            continue

        if label_func_name == '':
            label = labels_function(filename, 'cuda', output_folder)
        else:
            label = label_function(filename)
        if label is None:
            continue

        # Save dict for later
        data_entry = {"filename": filename, "label": label}

        all_final_json.append(data_dict)


    # Save Data in JSON file
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", type=str)
    args = parser.parse_args()

    with open(args.yaml, "r") as f:
        config = yaml.safe_load(f)

    for dataset_key, dataset_config in config.items():
        print(f"\n[{dataset_key}] 데이터셋 처리 중...")

        dataset_name = dataset_config["dataset_name"]
        src_dirs = dataset_config["src"]
        src_dirs = [src_dirs] if isinstance(src_dirs, str) else src_dirs
        dest_dir = dataset_config["dest"]
        use_keypoint = dataset_config.get("use_keypoint", True)

        label_func_name = dataset_config["label_function"]["name"]

        if label_func_name not in LABEL_REGISTRY:
            raise ValueError(f"라벨 함수 '{label_func_name}'가 레지스트리에 등록되지 않았습니다.")

        label_function = LABEL_REGISTRY[label_func_name]

        process_dataset(dataset_name, src_dirs, dest_dir, label_function, label_func_name, use_keypoint)
