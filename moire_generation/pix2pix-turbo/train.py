import os
import gc
import lpips
import clip
from tqdm.auto import tqdm
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats
import vision_aided_loss

from accelerate import Accelerator
from accelerate.utils import set_seed

import transformers
import diffusers
from diffusers.optimization import get_scheduler
from dataset import PairedDataset
from pix2pix_tb import Pix2Pix_Turbo
from utils import parse_args_paired_training

def cast_params(model, dtype):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            param.data = param.to(dtype)

def main(args):

    # api_key = os.getenv('WANDB_API_KEY')
    # wandb.login(key=api_key)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to
    )
    if accelerator.is_local_main_process:
        api_key = os.getenv('WANDB_API_KEY')
        wandb.login(key=api_key)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    net_pix2pix = Pix2Pix_Turbo(
        pretrained_diffusion_name=args.pretrained_model_name_or_path, 
        lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae,
        deterministic=args.deterministic)
    net_pix2pix.set_train()
    # cast_params(net_pix2pix, torch.float16)

    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())

    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)

    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_4.parameters())


    net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)
    net_lpips.eval() # 추가

    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()

    dataset_train = PairedDataset(
        path='/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/train',
        min_resize_res=768,
        max_resize_res=1024,
        crop_res=512,
        flip_prob=0.0,
        tokenizer=net_pix2pix.tokenizer
    )
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    
    dataset_val = PairedDataset(
        path='/purestorage/project/hkl/FAS_DnC/data/webdata/uhdm_data/test',
        min_resize_res=768,
        max_resize_res=1024,
        crop_res=512,
        flip_prob=0.0,
        tokenizer=net_pix2pix.tokenizer
    )
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.dataloader_num_workers)


    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)

    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)

    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    net_pix2pix, dl_train = accelerator.prepare(
        net_pix2pix, dl_train
    )
    net_clip, net_lpips, net_disc, optimizer, optimizer_disc, lr_scheduler, lr_scheduler_disc = accelerator.prepare(net_clip, net_lpips, net_disc, optimizer, optimizer_disc, lr_scheduler, lr_scheduler_disc)


    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_pix2pix.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
    
    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)


    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # if accelerator.is_main_process and args.track_val_fid:
    #     feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix, net_disc]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape

                x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)

                # Reconstruction loss
                # loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_l1 = F.l1_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                # loss = loss_l2 + loss_lpips
                loss = loss_l1 + loss_lpips

                # CLIP similarity loss
                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                    caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = (1 - clipsim.mean() / 100)
                    loss += loss_clipsim * args.lambda_clipsim

                accelerator.backward(loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)


                # # Generator
                # x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)
                # lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                # accelerator.backward(lossG)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                # optimizer.step()
                # lr_scheduler.step()
                # optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # # Discriminator
                # lossD_real = net_disc(x_tgt, for_real=True).mean() * args.lambda_gan
                # lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                # lossD = lossD_real + lossD_fake
                # accelerator.backward(lossD.mean())
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                # optimizer_disc.step()
                # optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                # lr_scheduler_disc.step()
                """
                Generator loss: fool the discriminator
                """
                x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)
                lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan
                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Discriminator loss: fake image vs real image
                """
                # real image
                lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                # fake image
                lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)
                lossD = lossD_real + lossD_fake

                

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    logs["lossG"] = lossG.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    # logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_l1"] = loss_l1.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    progress_bar.set_postfix(**logs)

                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(x_src[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]
                    
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)


                    # if global_step % args.eval_freq == 1:
                    #     l_l2, l_lpips, l_clipsim = [], [], []
                    #     if args.track_val_fid:
                    #         os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                    #     for step, batch_val in enumerate(dl_val):
                    #         if step >= args.num_samples_eval:
                    #             break
                    #         x_src = batch_val["conditioning_pixel_values"]
                    #         x_tgt = batch_val["output_pixel_values"]
                    #         B, C, H, W = x_src.shape
                    #         with torch.no_grad():
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
