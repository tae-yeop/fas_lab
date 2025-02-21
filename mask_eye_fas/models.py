import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import mediapipe as mp
LEFT_EYE_IDX = [
    33, 246, 161, 160, 159, 158, 157, 173,
    133, 155, 154, 153, 145, 144, 163, 7,
    468, 469, 470, 471  # 아이리스
]
RIGHT_EYE_IDX = [
    263, 466, 388, 387, 386, 385, 384, 398,
    362, 382, 381, 380, 374, 373, 390, 249,
    472, 473, 474, 475  # 아이리스
]

class LandmarkDetector():
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1, 
            min_detection_confidence=0.5,
            refine_landmarks=True 
        )

        self.landmark_points_68 = [
            162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
            296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
            380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87
        ]

    def forward(self, image):
        if image is None:
            return None
        
        height, width, _ = image.shape  
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                return face_landmarks.landmark

        return None
    
    def draw_landmarks(self, image, landmarks):
        for landmark in landmarks:
            cv2.circle(image, landmark, 2, (0, 255, 0), -1)
        return image

class SpoofChecker:
    def __init__(
        self,
        cfg,
        debug=False, 
        debug_prefix=None,
        output_dir=None
    ):

        self.cfg = cfg
        self.debug = debug
        self.debug_prefix = debug_prefix
        self.output_dir = output_dir
        
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if self.debug:
            self.edge_density_history = []
            self.shadow_diff_history = []
            self.reflection_ratio_history = []
            self.freq_mean_history = []
            self.flow_history = []


    def check_edge_pattern(self, eye_roi):
        """
        캐니 엣지 밀도 계산 
        """
        # get gray image
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

        # setup threshold
        use_otsu = self.cfg.get('use_otsu_for_canny', False)
        edge_density_thresh = self.cfg.get('edge_density_thresh', 0.1)

        if use_otsu:
            low_ratio = self.cfg.get('low_ratio', 0.5)

            ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            t2 = ret
            t1 = ret * low_ratio
        else:
            t1 = self.cfg.get('canny_threshold1', 50)
            t2 = self.cfg.get('canny_threshold2', 150)
    
        # get canny
        edges = cv2.Canny(gray, t1, t2)
        edge_density = np.mean(edges>0)
        suspicious = (edge_density > edge_density_thresh)

        if self.debug:
            output_name = f"{self.debug_prefix}_canny.jpg"
            cv2.imwrite(os.path.join(self.output_dir, output_name), edges)

        return suspicious, edge_density


    def check_shadow(self, eye_roi):
        """
        눈안 중앙이 어두운지 검사
        """
        # get gray image
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

        H, W = gray.shape
        cr = self.cfg.get('center_ratio', 0.5)
        sdiff = self.cfg.get('shadow_diff_thresh', -20)

        y1, y2 = int(H*(1-cr)/2), int(H*(1+cr)/2)
        x1, x2 = int(W*(1-cr)/2), int(W*(1+cr)/2)


        center_region = gray[y1:y2, x1:x2]
        mean_center = np.mean(center_region)
        mean_whole = np.mean(gray)

        # 차이 계산
        diff = mean_whole - mean_center

        shadow_susp = (diff < sdiff)
        
        if self.debug:
            output_name = f"{self.debug_prefix}_shadow.jpg"
            cv2.imwrite(os.path.join(self.output_dir, output_name), center_region)


        return shadow_susp, diff

    def check_reflection(self, eye_roi):
        """
        눈 안 반사 검사
        """
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

        # threshold
        ht = self.cfg.get('highlight_thresh', 140)
        min_r = self.cfg.get('min_ratio', 0.002)
        max_r = self.cfg.get('max_ratio', 0.2)

        mask = cv2.inRange(gray, ht, 255)
        ratio = np.sum(mask>0)/mask.size
        suspicious = (ratio < min_r or ratio > max_r)

        if self.debug:
            output_name = f"{self.debug_prefix}_reflection_mask.jpg"
            out_path = os.path.join(self.output_dir, output_name)
            cv2.imwrite(out_path, mask)

        return suspicious, ratio


    def analyze_frequency(self, eye_roi):
        """
        주파수 분석
        """
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

        # threshold
        freq_thresh = self.cfg.get('freq_thresh', 80.0)

        f32 = np.float32(gray)
        dft = cv2.dft(f32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        mag = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
        
        mag_spec = 20 * np.log(mag + 1e-8)

        freq_mean = np.mean(mag_spec)
        suspicious = (freq_mean > freq_thresh)

        return suspicious, freq_mean 

    def analyze_optical_flow(self, frames_eyes):
        """
        frames_eyes: list of grayscale eye_roi frames [gray1, gray2, ...]
        """

        frames_eyes = [f for f in frames_eyes if f is not None]

        if len(frames_eyes) < 2:
            return False, 0.0

        flow_range1 = self.cfg.get('flow_range1', 0.5)
        flow_range2 = self.cfg.get('flow_range2', 3.0)
        resize_dim = (64, 64)
        pyr_scale=0.5
        levels=3
        winsize=15
        iterations=3
        poly_n=5
        poly_sigma=1.2
        flags=0
        total_flow = 0.0
        count = 0

        for i in range(len(frames_eyes)-1):
            prev_frame = frames_eyes[i]
            next_frame = frames_eyes[i + 1]

            if prev_frame is None or next_frame is None:
                continue
            if prev_frame.size == 0 or next_frame.size == 0:
                continue


            if len(prev_frame.shape) == 3 and prev_frame.shape[2] == 3:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = prev_frame

            if len(next_frame.shape) == 3 and next_frame.shape[2] == 3:
                next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            else:
                next_gray = next_frame
            
            prev_gray = cv2.resize(prev_gray, resize_dim)
            next_gray = cv2.resize(next_gray, resize_dim)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, next_gray, None,
                pyr_scale, levels, winsize, iterations,
                poly_n, poly_sigma, flags
            )
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            mean_mag = np.mean(mag)
            total_flow += mean_mag
            self.flow_history.append(mean_mag)
            count += 1

            if self.debug:
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
                hsv[...,1] = 255
                hsv[...,0] = (ang*(180/np.pi/2)).astype(np.uint8)
                hsv[...,2] = np.clip(mag*10, 0, 255).astype(np.uint8)

                # HSV -> BGR
                bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                output_name = f"{self.debug_prefix}_flow_{i}.jpg"
                out_path = os.path.join(self.output_dir, output_name)
                cv2.imwrite(out_path, bgr_flow)

        if count == 0:
            return False, 0.0

        avg_flow = total_flow / count
        # suspicious = avg_flow 범위 벗어나면 True
        suspicious = (avg_flow < flow_range1 or avg_flow > flow_range2)
        return suspicious, avg_flow

    def extract_rppg_signal(self, frames_eye, method='green'):
        total_frames = len(frames_eye)
        rppg = []

        # setup threshold
        fs = 30
        low = 0.7
        high = 4.0
        method = 'green'

        for t in range(total_frames):
            if method == 'green':
                g_mean = np.mean(frames_eye[t][:, :, 1]) # B=0, G=1, R=2
                rppg.append(g_mean)
            else:
                avg = np.mean(frames_eye[t])
                rppg.append(avg)

        rppg = np.array(rppg, dtype=np.float32)

        freqs = np.fft.rfftfreq(total_frames, d=1/fs)  # 양의 주파수 스펙트럼
        fft_vals = np.fft.rfft(rppg)

        # bandpass
        for i, f in enumerate(freqs):
            if f < low or f > high:
                fft_vals[i] = 0

        # 역변환
        filtered_signal = np.fft.irfft(fft_vals, n=total_frames).astype(np.float32)
        

    
    
    def compute_lbp_feature(self, gray_img, neighbors=8, radius=1):
        """
        radius=1, neighbors=8 (8주변)
        radius=2, neighbors=16 (원형으로 더 넓은 범위)
        """
        H, W = gray_img.shape
        lbp = np.zeros((H, W), dtype=np.uint8)
        
        # 원형 이웃 좌표를 구해두는 로직 예시
        # (이해를 돕기 위한 pseudo 예시)
        offsets = []
        for n in range(neighbors):
            theta = 2.0 * np.pi * n / neighbors
            # 이웃 위치
            dx = radius * np.cos(theta)
            dy = radius * np.sin(theta)
            offsets.append((dx, dy))
        
        # 실제로는 보간(bilinear interpolation) 등 사용
        for r in range(radius, H - radius):
            for c in range(radius, W - radius):
                center_val = gray_img[r, c]
                code = 0
                for idx, (dx, dy) in enumerate(offsets):
                    # 이웃 픽셀 위치 (부동소수 -> 반올림)
                    nr = int(r + dy + 0.5)
                    nc = int(c + dx + 0.5)
                    neighbor_val = gray_img[nr, nc]
                    if neighbor_val > center_val:
                        code |= (1 << idx)
                lbp[r, c] = code

        return lbp


    def lbp_spoof_check(
        self, 
        eye_roi,
        lbp_ref_hist,
        neighbors=8, radius=1,
        chi_thresh=0.5,
        debug=False,
        debug_prefix=None
    ):
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

        lbp = compute_lbp_feature(gray, neighbors, radius)
        hist, _ = np.histogram(lbp, bins=2**neighbors, range=(0, 2**neighbors-1))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)

        ref = lbp_ref_hist + 1e-8
        cur = hist + 1e-8
        chi = np.sum((ref - cur)**2 / (ref + cur))
        suspicious = (chi > chi_thresh)

        if debug and debug_prefix is not None:
            # 1) LBP 맵 저장
            lbp_path = f"{debug_prefix}_lbp_map.jpg"
            cv2.imwrite(lbp_path, lbp)

            # 2) 히스토그램 시각화 (Matplotlib) -> 저장
            plt.figure()
            plt.bar(range(len(hist)), hist, width=0.8, color='blue', alpha=0.6, label='Current')
            plt.bar(range(len(lbp_ref_hist)), lbp_ref_hist, width=0.5, color='red', alpha=0.4, label='Reference')
            plt.title(f"LBP hist (chi={chi:.3f})")
            plt.legend()
            plt.savefig(f"{debug_prefix}_lbp_hist.png")
            plt.close()

        return suspicious, chi


    def process_frame(self, eye_roi):
        # Edge
        e_susp, e_val = self.check_edge_pattern(eye_roi)
        self.edge_density_history.append(e_val)

        # Shadow
        s_susp, s_diff = self.check_shadow(eye_roi)
        self.shadow_diff_history.append(s_diff)

        # Reflection
        r_susp, r_ratio = self.check_reflection(eye_roi)
        self.reflection_ratio_history.append(r_ratio)

        # Frequency
        f_susp, f_mean = self.analyze_frequency(eye_roi)
        self.freq_mean_history.append(f_mean)

        return e_susp, s_susp, r_susp, f_susp

    def plot_histories(self, prefix=None):
        frames = np.arange(len(self.edge_density_history))
        n_flow = len(self.flow_history)
        frames_flow = np.arange(n_flow)
        
        plt.figure(figsize=(12, 10)) 

        plt.subplot(3, 2, 1)
        plt.plot(frames, self.edge_density_history, 'r-', label="Edge Density")
        plt.title("Edge Density")
        plt.xlabel("Frame")
        plt.ylabel("Value")
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(frames, self.shadow_diff_history, 'g-', label="Shadow Diff")
        plt.title("Shadow Diff")
        plt.xlabel("Frame")
        plt.ylabel("Value")
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(frames, self.reflection_ratio_history, 'b-', label="Reflection Ratio")
        plt.title("Reflection Ratio")
        plt.xlabel("Frame")
        plt.ylabel("Value")
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(frames, self.freq_mean_history, 'm-', label="Frequency Mean")
        plt.title("Frequency Mean")
        plt.xlabel("Frame")
        plt.ylabel("Value")
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.plot(frames_flow, self.flow_history, 'c-', label="Optical Flow")
        plt.title("Optical Flow Magnitude")
        plt.xlabel("Frame")
        plt.ylabel("Value")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{prefix}_metrics_plot.png"))
        plt.close()

    