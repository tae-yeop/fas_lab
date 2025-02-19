import torch
import inspect

try:
    import bitsandbytes as bnb
    adam8bit_class = bnb.optim.Adam8bit
except ImportError:
    adam8bit_class = None
    # pass, raise ImportErro

try:
    import prodigyopt
    prodigy_class = prodigyopt.Prodigy
except ImportError:
    prodigy_class = None

optimizer_dict = {'adam': torch.optim.Adam, 'adam8bit': adam8bit_class, 'adamw': torch.optim.AdamW, 'prodigy': prodigy_class}

# def prepare_optimizer_params(models, learning_rates):
#     params_to_optimizer = []
#     for model, lr in zip(models, learning_rates):
#         model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
#         model_parameters_with_lr = {'params': model_parameters, 'lr':lr}
#         params_to_optimizer.append(model_parameters_with_lr)
#     return params_to_optimizer


def prepare_optimizer_params(model_module_dict, learning_rates):
    params_to_optimize = []
    for model_name, instance in model_module_dict.items():
        print('prepare_optimizer_params', type(instance))
        params_to_optimize.append({"params": list(filter(lambda p: p.requires_grad, instance.parameters())),
                                    "lr" : learning_rates[model_name]})
    return params_to_optimize