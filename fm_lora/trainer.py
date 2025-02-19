import torch

def lr_step_func(epoch, func_drop=[22, 30, 40]):
    return  0.1 ** len([m for m in func_drop if m - 1 <= epoch])

def get_scheduler(scheduler_type, optimizer_model, epoch, warmup, num_warmup_epochs, T_0, T_mult, eta_min, lr_func_drop, warmup_factor=1):
    lr_func = lambda epoch: lr_step_func(epoch, func_drop=lr_func_drop)

    if scheduler_type == "lambda":
        scheduler_model = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer_model,
            lr_lambda=lr_func
        )
    elif scheduler_type == "cosine":
        scheduler_model = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer_model,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
    else:
        raise ValueError()

    if warmup == True:
        scheduler_warmup_model = torch.optim.lr_scheduler.ConstantLR(
            optimizer_model,
            factor=warmup_factor,
            total_iters=num_warmup_epochs
        )

        scheduler_model = torch.optim.lr_scheduler.SequentialLR(
            optimizer_model,
            schedulers=[scheduler_warmup_model, scheduler_model],
            milestones=[num_warmup_epochs]
        )

    return scheduler_model


class Trainer():
    def __init__(
        self,
        model,
        transform,
        trainset, 
        dataloader, 
        training_type, 
        config, 
        fabric,
        header=None, 
        test_dataloader=None,
    )

        self.model = model
        self.header = header
        self.transform = transform
        self.trainset = trainset
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.training_type = training_type
        self.config = config

        self.start_epoch = 0
        self.global_step = self.config.global_step
        self.total_step = int(len(self.trainset) / config.batch_size / fabric.world_size * config.num_epoch)


class TrainerClip(Trainer):
    def __init__(
        self, 
    ):
        super().__init__()

    def start_training(self):
        if self.training_type == "text_image_contrastive":
            self.text_image_contrastive_training()
        elif self.training_type == "text_image_header":
            self.text_image_header_training()
        elif self.training_type == "image_encoder_only":
            self.image_encoder_only_training()
        elif self.training_type == "PAD_training":
            self.PAD_training()
        elif self.training_type == "PAD_training_only_header":
            self.PAD_training_only_header()
        else:
            raise ValueError()

    def text_image_contrastive_training(self):
        ...
        
    def PAD_training(self):
        # Optimizer
        optimizer_model = torch.optim.AdamW(
            params=[{'params': self.model.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_model, weight_decay=self.config.weight_decay
        )
        optimizer_header = torch.optim.AdamW(
            params=[{'params': self.header.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_header, weight_decay=self.config.weight_decay
        )


        scheduler_name = get_scheduler(
            scheduler_type=self.config.scheduler_type,
            optimizer_model=optimizer_header,
            epoch=self.config.num_epoch,
            warmup=self.config.warmup,
            num_warmup_epochs=self.config.num_warmup_epochs,
            T_0=self.config.T_0,
            T_mult=self.config.T_mult,
            eta_min=self.config.lr_header,
            lr_func_drop=self.config.lr_func_drop,
        )

        for epoch in range(self.start_epoch, self.config.num_epoch):
            for _, (images, target) in enumerate(self.dataloader):
                self.global_step += 1

                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                _, loss_image = self.header(F.normalize(self.model.module.encode_image(images)), target)

                loss_image.backward()

                clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_norm, norm_type=2)
                clip_grad_norm_(self.header.parameters(), max_norm=self.config.max_norm, norm_type=2)

                optimizer_model.step()
                optimizer_header.step()

                optimizer_model.zero_grad()
                optimizer_header.zero_grad()

            scheduler_model.step()
            scheduler_header.step()

    def PAD_training_only_header(self):
        # Freeze the clip model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        optimizer_header = torch.optim.AdamW(
            params=[{'params': self.header.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_header, weight_decay=self.config.weight_decay
        )

        