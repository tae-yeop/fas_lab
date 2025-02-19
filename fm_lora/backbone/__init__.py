from .clip_model import ClipModel

def get_model(cfg, device):

    if cfg.model_name= == 'clip':
        clip_model = ClipModel(
            cfg.model_name, device
        )

        return clip_model

    else:
        raise ValueError()


def get_output_dim(**kwargs):
    name = kwargs["model_name"]

    if name == 'clip':
        backbone_embeddings = {
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
            "ViT-L/14@336px": 768,
        }

        return backbone_embeddings[kwargs["backbone_size"]]
    else:
        raise ValueError()