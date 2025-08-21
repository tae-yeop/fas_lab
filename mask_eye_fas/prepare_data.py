# 데이터로 부터 각 메트릭에 대한 시계열을 뽑아내서 어디에 저장?
# /purestorage/datasets/NIA/Proj_115
import argparse
import cv2
import os
import numpy as np
import glob


from models import LandmarkDetector, SpoofChecker, LEFT_EYE_IDX, RIGHT_EYE_IDX
from utils import crop_eye_region

from dataclasses import dataclass, asdict,  field
from typing import List

@dataclass
class SpoofConfig:
    # canny edge thresholds
    use_otsu_for_canny:bool = False
    edge_density_thresh:float = 0.1
    low_ratio:float = 0.5
    canny_threshold1:int = 50
    canny_threshold2:int = 150

    # shadow check thresholds
    shadow_diff_thresh:float = -20
    center_ratio:float = 0.5

    # reflection check thresholds
    highlight_threshold:float = 140
    min_ratio:float = 0.002
    max_ratio:float = 0.2

    # frequency check thresholds
    freq_mean_thresh:float = 80.0

    # optical flow check thresholds
    flow_range1:float = 0.5
    flow_range2:float = 3.0

    # rppg check thresholds
    rppg_method:str = 'green'
    fps:int = 30
    bandpass_low:float = 0.7
    bandpass_high:float = 4.0
    rppg_ampl_thresh:float = 0.5
    skin_low:List[int] = field(default_factory=lambda: [0, 133, 77])
    skin_high:List[int] = field(default_factory=lambda: [235, 173, 127])
    snr_thresholds:List[int] = field(default_factory=lambda: [0.102, 0.125, 0.05, 0.07])

    # final spoof check thresholds
    total_count:int = 2

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

def save_features_npy():
    pass

def load_dataset_for_training(npy_dir):
    pass

def process_experiments(spoof_checker, root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_di, exist_ok=True)

    type_folders = [
        "attack_01_print_eye_flat",
        "attack_02_print_eye_curved",
        "attack_03_print_eye_nose_flat",
        "attack_04_print_eye_nose_curved",
        "attack_05_print_eye_nose_mouth_flat",
        "attack_06_print_eye_nose_mouth_curved",
        "real_01"
    ]

    for type_name in type_folders:
        color_dir = os.path.join(root_dir, type_name, "color")


def process_jpg_list(spoof_checker, detector, jpg_file_list, dataset_dir):

    aggregated_data = []
    for jpg_file in jpg_file_list:

        base_idx = jpg_file.find("Proj_115")
        base_idx2 = jpg_file.find("color")
        save_path = jpg_file[base_idx: base_idx2]

        save_path = os.path.join(dataset_dir, save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)



        img = cv2.imread(jpg_file)

        landmarks = detector.forward(img)

        if landmarks is None:
            continue

        left_eye_roi = crop_eye_region(img, landmarks, LEFT_EYE_IDX)
        right_eye_roi = crop_eye_region(img, landmarks, RIGHT_EYE_IDX)

        
        eye_left_feats = spoof_checker.extract_features(left_eye_roi)
        eye_right_feats = spoof_checker.extract_features(right_eye_roi)

        print(save_path)
        print(jpg_file, eye_left_feats)
        
        label = jpg_file.find('real')
        if (label == -1):
            label = 0 # fake
        else:
            label = 1 # real

        save_dict = {
            "filename" : jpg_file,
            "left_eye_feats": eye_left_feats,
            "right_eye_feats": eye_right_feats,
            "label" : label
        }

        aggregated_data.append(save_dict)

    npy_path = os.path.join(dataset_dir, "final_data.npy")
    np.save(npy_path, aggregated_data, allow_pickle=True)
    print(f"[DONE] Saved {len(aggregated_data)} items to {npy_path}.")


def process_jpg_list2(spoof_checker, detector, folder_list, dataset_dir):

    aggregated_data = []
    for folder in folder_list:

        left_eye_sequence = []
        right_eye_sequence = []

        label = folder.find('real')
        if (label == -1):
            label = 0 # fake
        else:
            label = 1 # real
        
        folder_dict = {
            'file_folder': folder,
            'label': label
        }
        eye_left_feats_dict = {
            'edge':[],
            'shadow':[],
            'reflection':[],
            'freq':[],
            'optical_flow':[],
        }
        eye_right_feats_dict = {
            'edge':[],
            'shadow':[],
            'reflection':[],
            'freq':[],
            'optical_flow':[],
        }

        for idx, file in enumerate(os.listdir(folder)):

            img_file_path = os.path.join(folder, file)
            img = cv2.imread(img_file_path)

            landmarks = detector.forward(img)

            if landmarks is None:
                continue

            left_eye_roi = crop_eye_region(img, landmarks, LEFT_EYE_IDX)
            right_eye_roi = crop_eye_region(img, landmarks, RIGHT_EYE_IDX)

            left_gray = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
            left_eye_sequence.append(left_gray)
            
            right_gray = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
            right_eye_sequence.append(right_gray)
            
            eye_left_feats = spoof_checker.extract_features(left_eye_roi)
            eye_right_feats = spoof_checker.extract_features(right_eye_roi)

            edge_l, shadow_l, refl_l, freq_l = eye_left_feats.values()
            edge_r, shadow_r, refl_r, freq_r = eye_right_feats.values()

            eye_left_feats_dict['edge'].append(edge_l)
            eye_left_feats_dict['shadow'].append(shadow_l)
            eye_left_feats_dict['reflection'].append(refl_l)
            eye_left_feats_dict['freq'].append(freq_l)

            eye_right_feats_dict['edge'].append(edge_r)
            eye_right_feats_dict['shadow'].append(shadow_r)
            eye_right_feats_dict['reflection'].append(refl_r)
            eye_right_feats_dict['freq'].append(freq_r)

        
        spoof_checker.analyze_optical_flow(left_eye_sequence)
        eye_left_feats_dict['optical_flow'].append(spoof_checker.flow_history)
        spoof_checker.flow_history = []


        spoof_checker.analyze_optical_flow(right_eye_sequence)
        eye_right_feats_dict['optical_flow'].append(spoof_checker.flow_history)

        folder_dict['eye_left_feats'] = eye_left_feats_dict
        folder_dict['eye_right_feats'] = eye_right_feats_dict

        aggregated_data.append(folder_dict)

    npy_path = os.path.join(dataset_dir, "final_data.npy")
    np.save(npy_path, aggregated_data, allow_pickle=True)
    print(f"[DONE] Saved {len(aggregated_data)} items to {npy_path}.")


if __name__ == '__main__':
    dataset_dir = "./output_npy"
    root_dir = '/purestorage/datasets/NIA/Proj_115'
    
    # pattern = os.path.join(root_dir, "*", "*", "*", "*", "color", "crop", "*.jpg")

    pattern = os.path.join(root_dir, "*", "*", "*", "*", "color", "crop")
    # # 한칸 내려가네
    # for f in os.listdir(root_dir):
    #     print(f)

    jpg_files = glob.glob(pattern)

    # 전체 갯수는 102630
    # /purestorage/datasets/NIA/Proj_115/0663/SR305/Light_02_Mid/attack_01_print_eye_flat/color/crop/010.jpg
    print("Found jpg files:", len(jpg_files))
    for f in jpg_files[:10]:  # 예시로 10개만 출력
        print(f)


    cfg = SpoofConfig().dict()
    detector = LandmarkDetector()
    checker = SpoofChecker(cfg, False)


    process_jpg_list2(checker, detector, jpg_files, dataset_dir)