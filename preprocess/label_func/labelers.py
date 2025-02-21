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