from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
import torch.nn as nn
import inspect

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
from training.objectives import loss_dict, gan_loss_dict
from models.architectures import arch_dict

def exists(x):
    return x is not None


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class BaseModel(ABC):
    def __init__(self, args):
        super().__init__()
        self.arg = args

    # @abstractmethod
    # def forward(self, x):
    #     pass

    def get_learnable_parameters(self):
        """
        모델의 모든 학습 가능한 파라미터를 리스트로 반환
        """
        model_parameters_dict = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                model_parameters_dict[name] = param
        return model_parameters_dict

    def _get_modules(self, modules, prefix, current_depth, max_depth):
        for name, module in self._modules.items():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(module, nn.Module) and current_depth < max_depth:
                modules[full_name] = module
                if current_depth < max_depth - 1:
                    module._get_modules(modules, full_name, current_depth + 1, max_depth)
                    
    def get_all_modules(self, depth=float('inf')):
        """
        모델의 모든 nn.Module을 OrderedDict로 반환
        """
        modules = OrderedDict()
        self._get_modules(modules, '', 0, depth)
        # for name, module in self.named_modules(depth=depth):
        #     if isinstance(module, nn.Module) and name != '':
        #         modules[name] = module
        return modules
    
    def set_requires_grad(self, module_name, requires_grad=True):
        """
        특정 모듈의 requires_grad 속성을 설정
        """
        if hasattr(self, module_name):
            module = getattr(self, module_name)
            for param in module.parameters():
                param.requires_grad = requires_grad
        else:
            print(f"Warning: {self.__class__.__name__} has no attribute {module_name}")

    def filter_valid_model_params(self, constructor, params_dict):
        valid_params = inspect.signature(constructor).parameters
        filtered_params = {key: value for key, value in params_dict.items() if key in valid_params}
        return filtered_params

    def set_up_optimization(self, args):
        pass



class Model1(BaseModel):
    """
    simple translator framework
    """
    def __init__(self, args, device):
        super().__init__(args)
        
        noiser_class = arch_dict[args.noiser_model['class']]
        noiser_model_params = self.filter_valid_model_params(noiser_class, args.noiser_model)
        self.noiser = noiser_class(**noiser_model_params).to(device)
        self.train_step = self.train_step_simple
        
        if exists(args.get('denoiser_model', None)) and exists(args.objective['noiser_loss'].get('cyc_loss', None)):
            denoiser_class = arch_dict[args.denoiser_model['class']]
            denoiser_model_params = self.filter_valid_model_params(denoiser_class, args.denoiser_model)
            self.denoiser = denoiser_class(**denoiser_model_params).to(device)
            self.train_denoiser = args.denoiser_model['training']
            if self.train_denoiser:
                self.train_step = self.train_step_multi
            else:
                print('args.denoiser_model', args.denoiser_model['pretrained'])
                denoiser_ckpt = torch.load(args.denoiser_model['pretrained'])
                self.denoiser.load_state_dict(denoiser_ckpt, strict=False)
                requires_grad(self.denoiser, False)
        else:
            self.denoiser = None
            
        if exists(args.get('discriminator_model', None)) and exists(args.objective['noiser_loss'].get('gan_loss', None)):
            discriminator_class = arch_dict[args.discriminator_model['class']]
            disc_model_params = self.filter_valid_model_params(discriminator_class, args.discriminator_model)
            self.disc = discriminator_class(**disc_model_params).to(device)
            self.train_step = self.train_step_gan
        else:
            self.disc = None

        print('denoiser 존재') if exists(self.denoiser) else None
        print('disc 존재') if exists(self.disc) else None
        
    def wrap_ddp(self, local_rank):
        print('local_rank2', local_rank)
        self.noiser = nn.parallel.DistributedDataParallel(self.noiser, device_ids=[local_rank], output_device=local_rank)
        self.denoiser = nn.parallel.DistributedDataParallel(self.denoiser, device_ids=[local_rank], output_device=local_rank) if self.denoiser and self.train_denoiser else self.denoiser
        self.disc = nn.parallel.DistributedDataParallel(self.disc, device_ids=[local_rank], output_device=local_rank) if exists(self.disc) else None
        self.models = [self.denoiser, self.noiser, self.disc]

    def unwrap_ddp(self):
        self.noiser = self.noiser.module
        self.denoiser = self.denoiser.module if exists(self.denoiser) and self.train_denoiser else None
        self.disc = self.disc.module if exists(self.disc) else None

    def get_module_in_ddp(self):
        module_dict = {'noiser': self.noiser.module}
        module_dict.update({'denoiser' : self.denoiser.module}) if exists(self.denoiser) and self.train_denoiser else None
        module_dict.update({'disc' : self.disc.module}) if exists(self.disc) else None
        return module_dict

    def train(self):
        self.noiser.train()
        self.denoiser.train() if exists(self.denoiser) else None
        self.disc.train() if exists(self.disc) else None

    def eval(self):
        self.noiser.eval()
        self.denoiser.eval() if exists(self.denoiser) else None
        self.disc.eval() if exists(self.disc) else None
        
    def set_objectives(self, objective_args, device):
        noiser_loss_args = objective_args['noiser_loss']

        noiser_single_perceptual_params = noiser_loss_args.get('single_vgg_loss', None)
        self.single_perceptual_coeff = noiser_single_perceptual_params['coeff'] if exists(noiser_single_perceptual_params) else None
        self.noiser_single_perceptual_loss = loss_dict['single_vgg_loss'](**noiser_single_perceptual_params['params']).to(device).eval() if exists(noiser_single_perceptual_params) else None
        
        noiser_multi_perceptual_params = noiser_loss_args.get('multi_perceptual', None)
        self.multi_perceptual_coeff = noiser_multi_perceptual_params['coeff'] if exists(noiser_multi_perceptual_params) else None
        self.noiser_multi_perceptual_loss = loss_dict['multi_vgg_loss'](**noiser_multi_perceptual_params['params']).to(device).eval() if exists(noiser_multi_perceptual_params) else None

        noiser_ffl_loss_params = noiser_loss_args.get('ffl', None)
        self.ffl_freq_coeff = noiser_ffl_loss_params['coeff'] if exists(noiser_ffl_loss_params) else None
        self.noiser_ffl_loss = loss_dict['ffl'](**noiser_ffl_loss_params['params']).to(device) if exists(noiser_ffl_loss_params) else None
        
        noiser_patch_freq_loss_params = noiser_loss_args.get('patch_freq', None)
        self.patch_freq_coeff = noiser_patch_freq_loss_params['coeff'] if exists(noiser_patch_freq_loss_params) else None
        self.noiser_patch_freq_loss = loss_dict['patch_freq'](**noiser_patch_freq_loss_params['params']).to(device) if exists(noiser_patch_freq_loss_params) else None

        noiser_cyc_loss_params = noiser_loss_args.get('cyc_loss', None)
        self.cyc_coeff = noiser_cyc_loss_params['coeff'] if exists(noiser_cyc_loss_params) else None
        self.noiser_cyc_loss = loss_dict['cyc_loss'](**noiser_cyc_loss_params['params']).to(device).eval() if exists(noiser_cyc_loss_params) else None
            
        noiser_gan_loss_params = noiser_loss_args.get('gan_loss', None)
        if exists(noiser_gan_loss_params) and exists(self.disc):
            gen_loss, disc_loss = gan_loss_dict[noiser_gan_loss_params['type']]
            self.noiser_gen_loss = gen_loss
            self.noiser_disc_loss = dics_loss
            if noiser_gan_loss_params['r1']:
                self.noiser_disc_r1_loss = gan_loss_dict['r1']
                self.r1_every = noiser_gan_loss_params['r1_every']
        else:
            self.noiser_gen_loss = None
            self.noiser_disc_loss = None
            self.noiser_disc_r1_loss = None
        
        
        if exists(self.denoiser) and self.train_denoiser:
            denoiser_loss_args = objective_args['denoiser_loss']
            denoiser_multi_perceptual_params = denoiser_loss_args.get('multi_perceptual', None)
            self.denoiser_multi_perceptual_loss = loss_dict['multi_vgg_loss'](**denoiser_multi_perceptual_params['params']).to(device).eval() if exists(denoiser_multi_perceptual_params) else None

            

    def forward(self, clean_img):
        noisy_hat1, _, _ = self.noiser(clean_img)
        return {'noisy': noisy_hat1}

    def train_step_simple(self, clean_img, noisy_img, total_step):
        n_loss = 0

        noisy_hat1, noisy_hat2, noisy_hat3 = self.noiser(clean_img)
        
        if exists(self.noiser_multi_perceptual_loss) : n_loss += self.noiser_multi_perceptual_loss(noisy_hat1, noisy_hat2, noisy_hat3, noisy_img, [0,1,2,3])
        if exists(self.noiser_ffl_loss) : n_loss += self.noiser_ffl_loss(noisy_img, noisy_hat1)
        if exists(self.noiser_patch_freq_loss) : n_loss += self.noiser_patch_freq_loss(noisy_img, noisy_hat1)
        if exists(self.noiser_cyc_loss):
            clean_cyc_hat1, clean_cyc_hat2, clean_cyc_hat3 = self.denoiser(noisy_hat1)
            n_loss += self.noiser_cyc_loss(clean_cyc_hat1, clean_cyc_hat2, clean_cyc_hat3, clean_img)
            
        return {'loss' : n_loss, 'noisy' : noisy_hat1}
    

    def train_step_gan(self, clean_img, noisy_img, total_step):
        pass
    #     n_loss = 0
    #     optimizer_g, optimizer_d = self.get_optimizers()
    #     noisy_hat1, noisy_hat2, noisy_hat3 = self.noiser(clean_img)
    #     # train D
    #     self.toggle_optimizer(optimizer_d)
    #     if total_step % self.r1_every == 0:
    #         optimizer_d.zero_grad()
    #         real_logit = self.disc()
    #     self.disc()
        
    def train_step_multi(self, clean_img, noisy_img, total_step):
        pass
    #     d_loss = 0
    #     n_loss = 0
    #     # assert self.denoiser_loss is not None, 'denoiser_loss must be specified'
    #     # assert self.noiser_loss is not None, 'noiser_loss must be specified'

    #     noisy_hat1, noisy_hat2, noisy_hat3 = self.noiser(clean_img)
    #     clean_hat1, clean_hat2, clean_hat3 = self.denoiser(noisy_img)

        
    #     if exists(self.noiser_multi_perceptual_loss) : n_loss += self.noiser_multi_perceptual_loss(noisy_hat1, noisy_hat2, noisy_hat3, noisy_img, [0,1,2,3])
    #     if exists(self.noiser_ffl_loss) : n_loss += self.noiser_ffl_loss(noisy_img, noisy_hat1)
    #     if exists(self.noiser_patch_freq_loss) : n_loss += self.noiser_patch_freq_loss(noisy_img, noisy_hat1)

    #     self.denoiser_multi_perceptual_loss

    #     clean_cyc_hat1, clean_cyc_hat2, clean_cyc_hat3 = self.denoiser(noisy_hat1)
        
    #     return {'noisy': noisy_hat1}
    def val_step(self, clean_img, noisy_img, total_step):
        return self.train_step(clean_img, noisy_img, total_step)

class Model1_WaveGAN(Model1):
    def train_step_simple(self, clean_img, noisy_img, total_step):
        n_loss = 0

        noisy_hat1 = self.noiser(clean_img)
        
        if exists(self.noiser_single_perceptual_loss) : n_loss += self.noiser_single_perceptual_loss(noisy_hat1, noisy_img, [0,1,2,3])
        if exists(self.noiser_ffl_loss) : n_loss += self.noiser_ffl_loss(noisy_img, noisy_hat1)
        if exists(self.noiser_patch_freq_loss) : n_loss += self.noiser_patch_freq_loss(noisy_img, noisy_hat1)
        # if exists(self.noiser_cyc_loss):
        #     clean_cyc_hat1, clean_cyc_hat2, clean_cyc_hat3 = self.denoiser(noisy_hat1)
        #     n_loss += self.noiser_cyc_loss(clean_cyc_hat1, clean_cyc_hat2, clean_cyc_hat3, clean_img)
            
        return {'loss' : n_loss, 'noisy' : noisy_hat1}

    def val_step(self, clean_img, noisy_img, total_step):
        return self.train_step(clean_img, noisy_img, total_step)

class Model2(BaseModel):
    """
    기존의 denoiser + noiser 같이 사용
    """
    def __init__(self, args, device):
        super().__init__(args)

        denoiser_class = arch_dict[args.denoiser_class]
        noiser_class = arch_dict[args.noiser_class]
        denoiser_model_params = self.filter_valid_model_params(denoiser_class, args.denoiser_model)
        noiser_model_params = self.filter_valid_model_params(noiser_class, args.noiser_model)

        self.denoiser = Denoiser(**denoiser_model_params).to(device)
        self.noiser = Denoiser(**noiser_model_params).to(device)

        # self.denoiser_loss = None
        # self.noiser_loss = None

    # def get_all_modules(self):
    #     return {'denoiser': self.}

    
    def wrap_ddp(self, local_rank):
        self.denoiser = nn.parallel.DistributedDataParallel(self.denoiser, device_ids=[local_rank], output_device=local_rank)
        self.noiser = nn.parallel.DistributedDataParallel(self.noiser, device_ids=[local_rank], output_device=local_rank)
        self.models = [self.denoiser, self.noiser]

    def get_module_in_ddp(self):
        return {'denoiser': self.denoiser.module, 
                'noiser': self.noiser.module}
    
    def unwrap_ddp(self):
        self.denoiser = self.denoiser.module
        self.noiser = self.noiser.module

    def set_objectives(self, objective_args, device):
        denoiser_loss_args = objective_args['denoiser_loss']
        noiser_loss_args = objective_args['noiser_loss']

        denoiser_multi_perceptual_params = denoiser_loss_args.get('multi_perceptual', None)

        noiser_multi_perceptual_params = noiser_loss_args.get('multi_perceptual', None)
        noiser_ffl_loss_params = noiser_loss_args.get('ffl', None)
        noiser_patch_freq_loss_params = noiser_loss_args.get('patch_freq', None)

        self.denoiser_multi_perceptual_loss = loss_dict['multi_vgg_loss'](**denoiser_multi_perceptual_params['params']).to(device).eval() if exists(denoiser_multi_perceptual_params) else None
        # if exists(denoiser_multi_perceptual_params):
        #     self.denoiser_multi_perceptual_loss = loss_dict['multi_vgg_loss'](**denoiser_multi_perceptual_params['params']).to(device)
        # else:
        #     self.denoiser_multi_perceptual_loss = None

        self.noiser_multi_perceptual_loss = loss_dict['multi_vgg_loss'](**noiser_multi_perceptual_params['params']).to(device).eval() if exists(noiser_multi_perceptual_params) else None
        # if exists(noiser_multi_perceptual_params):
        #     self.noiser_multi_perceptual_loss = loss_dict['multi_vgg_loss'](**noiser_multi_perceptual_params['params']).to(device)
        # else:
        #     self.noiser_multi_perceptual_loss = None

        self.noiser_ffl_loss = loss_dict['ffl'](**noiser_ffl_loss_params['params']).to(device) if exists(noiser_ffl_loss_params) else None
        # if exists(noiser_ffl_loss_params):
        #     self.noiser_ffl_loss = loss_dict['ffl'](**noiser_ffl_loss_params['params']).to(device)
        # else:
        #     self.noiser_ffl_loss = None

        self.noiser_patch_freq_loss = loss_dict['patch_freq'](**noiser_patch_freq_loss_params['params']).to(device) if exists(noiser_patch_freq_loss_params) else None
        # if exists(noiser_patch_freq_loss_params):
        #     self.noiser_patch_freq_loss = loss_dict['patch_freq'](**noiser_patch_freq_loss_params['params']).to(device)
        # else:
        #     self.noiser_patch_freq_loss = None
        
        # self.denoiser_loss = loss_dict[denoiser_loss_args['type']](denoiser_loss_args['params'], device)
        # self.noiser_loss = loss_dict[noiser_loss_args['type']](noiser_loss_args['params'], device)

        # for loss_info in denoiser_loss:
        #     loss_dict[loss_info['type']](**loss_info['params'])

    def train(self):
        self.denoiser.train()
        self.noiser.train()

    def eval(self):
        self.denoiser.eval()
        self.noiser.eval()

        
    def forward(self, clean_img, noisy_img):
        # d_loss = 0
        # n_loss = 0
        # # with ground truth
        # clean_hat1, clean_hat2, clean_hat3 = self.denoiser(noisy_img)
        # noisy_hat1, noisy_hat2, noisy_hat3 = self.noiser(clean_img)

        # if exists(self.denoiser_multi_perceptual_loss):
        #     self.denoiser_multi_perceptual_loss.eval()
        #     print('training', self.denoiser_multi_perceptual_loss.loss_fn.blocks[0].training)
        #     d_loss += self.denoiser_multi_perceptual_loss(clean_hat1, clean_hat2, clean_hat3, clean_img, [0,1,2,3])
        # if exists(self.noiser_multi_perceptual_loss):
        #     self.noiser_multi_perceptual_loss.eval()
        #     print('training', self.noiser_multi_perceptual_loss.loss_fn.blocks[0].training)
        #     n_loss += self.noiser_multi_perceptual_loss(noisy_hat1, noisy_hat2, noisy_hat3, noisy_img, [0,1,2,3])
        # if exists(self.noiser_ffl_loss):
        #     n_loss += self.noiser_ffl_loss(noisy_img, noisy_hat1)
        # if exists(self.noiser_patch_freq_loss):
        #     n_loss += self.noiser_patch_freq_loss(noisy_img, noisy_hat1)

        # return {'loss' : d_loss + n_loss, 'clean': clean_hat1, 'noisy' : noisy_hat1}

        clean_hat1, clean_hat2, clean_hat3 = self.denoiser(noisy_img)
        noisy_hat1, noisy_hat2, noisy_hat3 = self.noiser(clean_img)
        # noisy_hat1, noisy_hat2, noisy_hat3 = self.noiser(clean_img)
        return {'clean': clean_hat1, 'noisy' : noisy_hat1}

    def train_step(self, clean_img, noisy_img):
        d_loss = 0
        n_loss = 0

        clean_hat1, clean_hat2, clean_hat3 = self.denoiser(noisy_img)
        noisy_hat1, noisy_hat2, noisy_hat3 = self.noiser(clean_img)

        if exists(self.denoiser_multi_perceptual_loss) : d_loss += self.denoiser_multi_perceptual_loss(clean_hat1, clean_hat2, clean_hat3, clean_img, [0,1,2,3])

        if exists(self.noiser_multi_perceptual_loss) : n_loss += self.noiser_multi_perceptual_loss(noisy_hat1, noisy_hat2, noisy_hat3, noisy_img, [0,1,2,3])
        if exists(self.noiser_ffl_loss) : n_loss += self.noiser_ffl_loss(noisy_img, noisy_hat1)
        if exists(self.noiser_patch_freq_loss) : n_loss += self.noiser_patch_freq_loss(noisy_img, noisy_hat1)
        
        return {'loss' : d_loss + n_loss, 'clean': clean_hat1, 'noisy' : noisy_hat1}

    def val_step(self, clean_img, noisy_img):
        return self.train_step(clean_img, noisy_img)
     
# class Model3(BaseModel):
#     """
#     Noise feature + Clean image separation
#     """
#     def __init__(self, args):
#         super().__init__(args)
#         self.cfgs = args
#         self.separator = ...
#         self.geneartor = ...
#         if cfgs.denoiser:
#             self.denoiser = ...
#         else:
#             self.denoiser = None
#         if cfgs.discriminator:
#             self.discriminator = ...
#         else:
#             self.discriminator = None
#         self.loss = ...
#         self.loss1 = nn.SmoothL1Loss()
#         self.loss2 = PerceptualLoss()

#     def forward(self, clean_img, noisy_img):
#         noise_feature = self.extractor(noisy_img)
#         noisy_hat = self.geneartor(torch.concat([noise_feature, clean_img]))
#         clean_hat = self.denoiser(noisy_hat)
#         return clean_hat, noisy_hat, noise_feature

#     def calc_loss(self, clean_img, noisy_img):
#         total_loss = 0
#         clean_hat, noisy_hat, noise_feature = self.forward(clean_img, noisy_img)
#         total_loss += self.loss1(clean_img, clean_hat)
#         self.loss2(clean_img, clean_hat)


#         self.discriminator()

# class Model2(BaseModel):
#     """
#     Noiser with Diffusion Training
#     """
#     def __init__(self, ):
#         super().__init__()
#         ...

# class Model3(BaseModel):
#     """
#     Noiser with Diffusion Training
#     """
#     def __init__(self, ):
#         super().__init__()
#         ...



model_dict = {
    'model1': Model1,
    'model1_wg': Model1_WaveGAN,
    'model2': Model2,
    # 'model3': Model3
}