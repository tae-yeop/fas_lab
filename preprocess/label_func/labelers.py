import mediapipe as mp
import torch

class LandmarkDetector():
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1, 
                min_detection_confidence=0.5,
                refine_landmarks=True 
            )
        
        return cls._instance

    @classmethod
    def forward(cls, image):
        if image is None:
            return None

        results = cls._instance.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                return face_landmarks.landmark

        return None

class DepthEstimator():
    _instance = None
    _transform = None

    @classmethod
    def get_instance(cls, model_type="DPT_Large"):
        if cls._instance is None:
            cls._instance, cls._transform = self.get_midas(model_type)
        return cls._instance, cls._transform


    def get_midas(self, model_type="DPT_Large"):
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.eval()
        midas_transforms =  torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        return midas, transform


class BlinkEyeFeatureExtractor():
    _instance = None
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.edge_density_history = []
        self.shadow_diff_history = []
        self.reflection_ratio_history = []
        self.freq_mean_history = []
        self.flow_history = []

    @classmethod
    def get_instance(cls, cfg):
        if cls._instance is None:
            cls._instance = self.__init__(cfg)
        return cls._instance
            
    def get_edge(self, eye_roi):
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        t1 = self.cfg.get('canny_threshold1', 50)
        t2 = self.cfg.get('canny_threshold2', 150)

        edges = cv2.Canny(gray, t1, t2)
        edge_density = np.mean(edges>0)

        return edge_density

    def get_shadow(self, eye_roi):
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

        H, W = gray.shape
        cr = self.cfg.get('center_ratio', 0.5)
        sdiff = self.cfg.get('shadow_diff_thresh', -20)

        y1, y2 = int(H*(1-cr)/2), int(H*(1+cr)/2)
        x1, x2 = int(W*(1-cr)/2), int(W*(1+cr)/2)


        center_region = gray[y1:y2, x1:x2]
        mean_center = np.mean(center_region)
        mean_whole = np.mean(gray)

        diff = mean_whole - mean_center

        return diff

    def get_reflection(self, eye_roi):
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

        # threshold
        ht = self.cfg.get('highlight_thresh', 140)
        min_r = self.cfg.get('min_ratio', 0.002)
        max_r = self.cfg.get('max_ratio', 0.2)

        mask = cv2.inRange(gray, ht, 255)
        ratio = np.sum(mask>0)/mask.size

        return ratio

    def get_frequency(self, eye_roi):
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)

        # threshold
        freq_thresh = self.cfg.get('freq_thresh', 80.0)

        f32 = np.float32(gray)
        dft = cv2.dft(f32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        mag = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
        
        mag_spec = 20 * np.log(mag + 1e-8)

        freq_mean = np.mean(mag_spec)

        return freq_mean

    @classmethod
    def process_frame(cls, eye_roi):
        e_val = self.get_edge(eye_roi)
        self.edge_density_history.append(e_val)

        s_diff = self.get_shadow(eye_roi)
        self.shadow_diff_history.append(s_diff)

        r_ratio = self.get_reflection(eye_roi)
        self.reflection_ratio_history.append(r_ratio)

        f_mean = self.get_frequency(eye_roi)
        self.freq_mean_history.append(f_mean)

        return {
            'edge': e_val,
            'shadow': s_diff,
            'reflection': r_ratio,
            'frequency': f_mean
        }
