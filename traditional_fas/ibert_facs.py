import cv2
import torch
import numpy as np
from libreface import LibreFace

# LibreFace 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LibreFace(device=device)

# 비디오 로드 (종이 얼굴 vs 실제 얼굴)
video_path_paper = "paper_face.mp4"  # 종이 얼굴 비디오
video_path_real = "real_face.mp4"  # 실제 얼굴 비디오

# OpenCV를 사용하여 비디오 프레임 처리
def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    au_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # LibreFace는 RGB 입력 필요

        # LibreFace로 AU 분석
        results = model.infer(frame_rgb)

        if results:
            au_vector = results[0]['au']  # Action Units 값 추출
            au_features.append(au_vector)
        
        # 실시간 프레임 표시 (디버깅용)
        cv2.imshow("Video Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return np.array(au_features)

# 종이 얼굴 vs 실제 얼굴 비교
print("Processing Paper Face Video...")
au_paper = process_video(video_path_paper, model)
print("Processing Real Face Video...")
au_real = process_video(video_path_real, model)

# AU 값의 표준 편차(움직임 분석)
paper_std = np.std(au_paper, axis=0)
real_std = np.std(au_real, axis=0)

# 스푸핑 탐지 기준 설정
threshold = 0.02  # 표정 변화 감지를 위한 임계값 (수정 가능)

# AU 변화량이 낮을 경우(=거의 움직이지 않음) 종이 얼굴로 판단
paper_score = np.mean(paper_std)
real_score = np.mean(real_std)

print(f"Paper Face AU Variability Score: {paper_score:.4f}")
print(f"Real Face AU Variability Score: {real_score:.4f}")

if paper_score < threshold:
    print("❌ 종이 얼굴(스푸핑 공격) 감지됨!")
else:
    print("✅ 정상적인 얼굴 (실제 사람)")

if real_score < threshold:
    print("❌ 의심스러운 비디오! 하지만 실제 얼굴일 가능성이 높음")
else:
    print("✅ 정상적인 얼굴 (실제 사람)")