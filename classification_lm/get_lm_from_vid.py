import cv2
import numpy as np
import mediapipe as mp
import argparse
import os

class LandmarkDetector:
    def __init__(self, landmark_model="68"):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1, 
            min_detection_confidence=0.5
        )

        if landmark_model == "68":
            self.landmark_points = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                    296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                    380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
        else:
            self.landmark_points = None
    def forward(self, image):
        height, width, _ = image.shape
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = []
                for idx in self.landmark_points:
                    landmark = face_landmarks.landmark[idx]
                    landmarks.append((int(landmark.x * width), int(landmark.y * height)))
                return landmarks
            
        return None
    

# class Video

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='', help='Video file path')
    parser.add_argument('--output', type=str, default='', help='Output video file path')
    parser.add_argument('--fps', type=int, default=30, help='FPS of output video')
    
    args = parser.parse_args()

    detector = LandmarkDetector()

    cap = cv2.VideoCapture(args.video if args.video else 0)

    ret, frame = cap.read()
    if not ret:
        print("Failed to open video.")
        cap.release()
        exit()

    # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame_size = (frame.shape[1], frame.shape[0]) # (1920, 1080)
    print('frame size', frame_size)
    # height, width = frame.shape[:2]
    # frame_size = (width, height) 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 너비 : 1920, 높이: 1080으로 저장됨
    video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (1080, 1920))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print("Original frame shape:", frame.shape)  # (Height, Width, Channels)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print("Rotated frame shape:", frame.shape) 
        landmarks = detector.forward(frame)
        if landmarks:
            for landmark in landmarks:
                cv2.circle(frame, landmark, 2, (0, 255, 0), -1)
        video_writer.write(frame)

    cap.release()
    video_writer.release()

