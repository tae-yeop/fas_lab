
import argparse
import cv2
import os

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spoof Detection')
    # /purestorage/AILAB/AI_1/tyk/3_CUProjects/iBeta/dataset/sample_vids/notebook/exps/real.mp4
    parser.add_argument('--input_video_path', type=str, default='/purestorage/AILAB/AI_1/tyk/3_CUProjects/iBeta/dataset/sample_vids/notebook/exps/mask.mp4')
    parser.add_argument('--output_dir', type=str, default='/purestorage/AILAB/AI_1/tyk/3_CUProjects/fas_lab/eye_crop_fas/output2')
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--frame_size', type=int, default=640)

    args = parser.parse_args()

    cfg = SpoofConfig().dict()
    detector = LandmarkDetector()
    checker = SpoofChecker(cfg, args.debug, output_dir=args.output_dir)

    cap = cv2.VideoCapture(args.input_video_path)
    file_name = os.path.splitext(os.path.basename(args.input_video_path))[0]

    left_eye_sequence = []   # 왼눈 시퀀스(그레이)
    right_eye_sequence = []  # 오른눈 시퀀스(그레이)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_size = (args.frame_size, args.frame_size)
        frame = cv2.resize(frame, frame_size)

        landmarks = detector.forward(frame)
        if landmarks is None:
            continue

        left_eye_roi = crop_eye_region(frame, landmarks, LEFT_EYE_IDX)
        right_eye_roi = crop_eye_region(frame, landmarks, RIGHT_EYE_IDX)
        
        checker.debug_prefix = file_name+str(idx)
        checker.process_frame(left_eye_roi)

        left_gray = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
        left_eye_sequence.append(left_gray)

        right_gray = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)
        right_eye_sequence.append(right_gray)

        idx += 1

    cap.release()
    cv2.destroyAllWindows()


    checker.analyze_optical_flow(left_eye_sequence)
    checker.plot_histories('left')