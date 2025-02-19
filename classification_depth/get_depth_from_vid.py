import torch
import cv2
import numpy as np
import os
import argparse


class DepthEstimator:
    def __init__(self, model_type="DPT_Large"):
        self.model, self.transform = self.get_depth_model(model_type)

    def get_depth_model(self, model_type):
        if model_type == "DPT_Large":
            model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        elif model_type == "DPT_Hybrid":
            model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        else:
            raise ValueError(f"Model type {model_type} not recognized")
        
        model.cuda()
        model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform

        return model, transform
    
    def forward(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = self.transform(image).to("cuda") # [1, 3, 384, 672]
        with torch.no_grad():
            prediction = self.model(image) # [1, 384, 672]
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(image.shape[2], image.shape[3]),
                mode="bicubic",
                align_corners=False,
            ).squeeze() # (384, 672)
        output = prediction.cpu().numpy()
        return output


def fix_rotation(frame):
    """이미지가 시계 방향으로 90도 회전했다면 복구"""
    height, width = frame.shape[:2]
    if width > height:  # 가로가 더 길다면, 90도 회전되어 있음
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame  # 정상적이면 그대로 반환

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="", help="Video file path")
    parser.add_argument("--output", type=str, default="", help="Output video file path")
    parser.add_argument('--fps', type=int, default=30, help='FPS of output video')

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video if args.video else 0)

    ret, frame = cap.read()
    if not ret:
        print("Failed to open video.")
        cap.release()
        exit()
    
    frame_size = (frame.shape[0], frame.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (640, 640))

    model = DepthEstimator()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = fix_rotation(frame)
        depth_map = model.forward(frame)

        print('depth map shape:', depth_map.shape)  # (3,)
        print('frame shape:', frame.shape)  # (1080, 1920, 3)
        # Depth Map을 정규화하고 컬러맵 적용
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        depth_colored_resized = cv2.resize(depth_colored, (640, 640))
        frame_resized = cv2.resize(frame, (640, 640))
        # Depth Map을 원본 영상 위에 오버레이 (50% 투명도 적용)
        blended_frame = cv2.addWeighted(frame_resized, 0.5, depth_colored_resized, 0.5, 0)

        video_writer.write(blended_frame)

    cap.release()
    video_writer.release()