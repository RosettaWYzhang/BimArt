import os
import sys
sys.path.append("../utils")
sys.path.append("../model_arch")
from model_arch.contact_transformer import ContactTransformer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from utils import model_util
from diffusers.optimization import get_scheduler
import torch
from dataset.contact_data import ObjectContactData


def load_contact_dataset(cfg):
    train_dataset = None
    train_dataset = ObjectContactData(split="train",
                                      base_dir=cfg["base_dir"],
                                      end_frame=cfg["end_frame"],
                                      pred_horizon=cfg["pred_horizon"],
                                      return_aux_info=False) 
    return train_dataset

def load_contact_module(cfg, device, train_dataloader=None):
    encoder_input_dim = cfg["num_bps_points"] * cfg["num_features"] 
    input_dim = cfg["num_bps_points"] * 2
    prev_input_dim = cfg["num_bps_points"] * (cfg["num_features"] + 2)
    # part based, dim x 2
    input_dim *= 2
    prev_input_dim *= 2
    encoder_input_dim *= 2
    global_state_dim = cfg["global_state_dim"]
    contact_model = ContactTransformer(input_feats=input_dim,
                                        latent_dim=cfg["latent_cond_dim"],
                                        ff_size=cfg["ff_size"],
                                        num_layers=cfg["num_layers"],
                                        num_heads=cfg["num_heads"],
                                        dropout=cfg["dropout"],
                                        activation=cfg["activation"],
                                        bps_input_dim=encoder_input_dim + global_state_dim,
                                        pred_horizon=cfg["pred_horizon"],
                                        diffusion_step_embed_dim=cfg["latent_cond_dim"],
                                        )


    contact_model.to(device)

    if train_dataloader is not None:
        optimizer = torch.optim.AdamW(
        params=contact_model.parameters(),
        lr=1e-5, weight_decay=1e-6)
        # Cosine LR schedule with linear warmup
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=len(train_dataloader) * cfg["num_epochs"]
    )
    else:
        optimizer = None
        lr_scheduler = None

    if cfg["load_pretrained"]:
        model_file = os.path.join(cfg["save_root_dir"], cfg["exp_name"], cfg["pretrained_model_path"])
        if os.path.exists(model_file):
            print("--- loading pretrained model from --- " + model_file, flush=True)
            contact_model, _, optimizer, lr_scheduler, start_epoch = model_util.load_model_optimizer_lrscheduler_checkpt(
                contact_model, optimizer, lr_scheduler, model_path=model_file)
            print("loaded model at epoch: ", start_epoch)
            cfg["start_epochs"] = start_epoch + 1
            print("*****************updating starte epoch to :  %d" %cfg["start_epochs"])
        else:
            print("pretrained contact model not found, retraining a new model")
    return contact_model, optimizer, lr_scheduler


def load_noise_scheduler(cfg):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["num_timesteps"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=False, 
        prediction_type="sample",
    )
    return noise_scheduler



