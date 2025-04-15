import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from model_arch.motion_transformer import MotionTransformer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
import torch



def createNN(cfg, device):
    contact_dim = cfg["data"]["num_bps_points"] * 4 # 2x part-wise bps, 2x separate left and right hand
    global_dim = cfg["data"]["global_state_dim"]
    bps_input_dim = cfg["data"]["bps_feature_dim"] * cfg["data"]["num_bps_points"] * 2 + global_dim
    base_model = MotionTransformer(input_feats=cfg["data"]["action_dim"], 
                        ff_size=cfg["model"]["ff_size"], 
                        num_layers=cfg["model"]["num_layers"],
                        num_heads=cfg["model"]["num_heads"], 
                        dropout=cfg["model"]["dropout"], 
                        activation=cfg["model"]["activation"],
                        latent_dim=cfg["model"]["latent_cond_dim"], 
                        bps_input_dim=bps_input_dim, 
                        pred_horizon=cfg["data"]["pred_horizon"],
                        contact_input_dim=contact_dim, 
                        diffusion_step_embed_dim=cfg["model"]["diffusion_step_embed_dim"])

    return base_model.to(device)


def createScheduler(cfg):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["train"]["num_diffusion_iters"],
        beta_schedule=cfg["model"]["beta"],
        clip_sample=False, 
        prediction_type="sample",
    )
    return noise_scheduler


def create_ema_model(base_model):
    ema = EMAModel(
        parameters=base_model.parameters(),
        power=0.75)
    return ema


def create_optimizer(base_model, lr):
    optimizer = torch.optim.AdamW(params=base_model.parameters(),lr=lr, weight_decay=1e-6)
    return optimizer


def create_lr_scheduler(optimizer, warmup_steps, dataloader_len, num_epochs):
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps, 
        num_training_steps=dataloader_len * num_epochs
    )
    return lr_scheduler 


def save_model_optimizer_lrscheduler_checkpt(model, epoch_idx, optimizer, lr_scheduler, model_path="model.pth", ema_model=None):
    # save model
    checkpoint = { 
        'epoch': epoch_idx,
        'model': model.state_dict(),
        'ema_stat_dict': ema_model.state_dict() if ema_model is not None else None,
        'optimizer': optimizer.state_dict(),
        'lr_sched': lr_scheduler.state_dict() if lr_scheduler is not None else None}
    torch.save(checkpoint, model_path)


def load_model_optimizer_lrscheduler_checkpt(model, optimizer=None, lr_scheduler=None, model_path="model.pth", ema_model=None):
    '''
    Returns: 5 objects:
      model, ema_model(default None), optimizer, lr_scheduler, epoch
    '''
    print("********************************* trying to load model from checkpoint: ", model_path)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    print("************************************* successfully loaded model from checkpoint *************************************")
    if ema_model is not None:
        ema_model.load_state_dict(checkpoint['ema_stat_dict'] if checkpoint['ema_stat_dict'] is not None else None)
        ema_model.shadow_params = [p.clone().detach() for p in model.parameters()]
    epoch = checkpoint['epoch']
    return model, ema_model, optimizer, lr_scheduler, epoch


