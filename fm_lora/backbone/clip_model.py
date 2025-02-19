import clip

class ClipModel():
    def __init__(self, model_name, device):
        self.backbone, _ = clip.load(model_name, device, jit=False)