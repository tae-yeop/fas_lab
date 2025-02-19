


def get_header():
    loss = kwargs["loss"]

    if loss == "ElasticArcFace":
        header = ElasticArcFace(in_features=backbone_out_dim, out_features=kwargs["num_classes"],
                                s=kwargs["s"], m=kwargs["m"], std=kwargs["std"])
    elif loss == "ElasticArcFacePlus":
        header = ElasticArcFace(in_features=backbone_out_dim, out_features=kwargs["num_classes"],
                                s=kwargs["s"], m=kwargs["m"], std=kwargs["std"], plus=plus)
    elif loss == "ElasticCosFace":
        header = ElasticCosFace(in_features=backbone_out_dim, out_features=kwargs["num_classes"],
                                s=kwargs["s"] , m=kwargs["m"], std=kwargs["std"])
    elif loss == "ElasticCosFacePlus":
        header = ElasticCosFace(in_features=backbone_out_dim, out_features=kwargs["num_classes"],
                                s=kwargs["s"] , m=kwargs["m"], std=kwargs["std"], plus=plus)
    elif loss == "ArcFace":
        header = ArcFace(in_features=backbone_out_dim, out_features=kwargs["num_classes"],
                         s=kwargs["s"], m=kwargs["m"])
    elif loss == "CosFace":
        header = CosFace(in_features=backbone_out_dim, out_features=kwargs["num_classes"],
                         s=kwargs["s"] , m=kwargs["m"])
    elif loss == "AdaFace":
        header = AdaFace(embedding_size=backbone_out_dim, classnum=kwargs["num_classes"])
    elif loss == "BinaryCrossEntropy":
        header = BinaryCrossEntropyHeader(in_features=backbone_out_dim, out_features=kwargs["num_classes"])
    else:
        raise ValueError()