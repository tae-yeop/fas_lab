from typing import Optional
import json
import os
import cv2
import torch

from .registry import register_label_function
from .labelers import LandmarkDetector, DepthEstimator

# 68개 랜드마크 인덱스
LANDMARK_POINTS_68 = [
    162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,
    454,389,71,63,105,66,107,336,296,334,293,301,168,197,5,4,
    75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
    380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,
    308,317,14,87
]


@register_label_function("test_label")
def test_label_function(filename: str) -> Optional[int]:
    if "real" in filename:
        return 1
    elif "print" in filename:
        return 0
    return None

@register_label_function("fas_paper_label")
def fas_paper_label_function(filename):
    """
    label for dataset fas_paper
    """
    label = filename.find('live')
    if (label == -1):
        return 0
    else:
        return 1

@register_label_function("mp_landmarks")
def label_with_landmarks(filename: str):
    image = cv2.imread(filename)
    if image is None:
        return None
    
    face_mesh = LandmarkDetector.get_instance()
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            return face_landmarks.landmark

    return None

@register_label_function("mp_landmarks_68")
def label_with_landmarks_68(filename: str):
    image = cv2.imread(filename)
    if image is None:
        return None
    
    face_mesh = LandmarkDetector.get_instance()
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for idx in LANDMARK_POINTS_68:
                landmark = face_landmarks.landmark[idx]
                landmarks.append((int(landmark.x * width), int(landmark.y * height)))
            return landmkars

    return None


@register_label_function("depth_midas")
def label_with_depth_map_midas(filename: str, device: str, dest_path: str):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    estimator, transform = DepthEstimator.get_instance()
    estimator = estimator.to(device)
    
    image_input = transform(image).to(device)

    with torch.no_grad():
        depth = estimator(image_input)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()


    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    cv2.imwrite(dest_path, depth_colored)

    return depth_colored