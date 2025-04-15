import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from torch import nn
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils import data_util
from utils import yaml_util, model_util
import datetime
import argparse
import wandb
from contact_prior.contact_prior_util import load_contact_module, load_noise_scheduler, load_contact_dataset
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda")
torch.set_printoptions(precision=10)
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--yaml_file', type=str, default="config_files/train_contact_config.yaml")
args = parser.parse_args()
print(args)


def train(save_dir, dataloader, noise_scheduler, model, optimizer, lr_scheduler, start_epochs, num_epochs, use_wandb=False):

    for epoch_idx in range(start_epochs, start_epochs + num_epochs):
        loss_acc = 0
        epoch_start_time = datetime.datetime.now()
        # batch loop
        for bn, nbatch in enumerate(dataloader): # debug use the same nbatch
            nbatch_norm = data_util.preprocess_contact_batch(nbatch, device, stat_dict, cfg["normalize_data"]) # convert to float tensor and cuda device
            naction = nbatch_norm['action'].to(device).float().clone()
            naction_gt = nbatch_norm['action'].to(device).float().clone()
            noise = torch.randn(naction.shape, device=device).float()

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (nbatch_norm['action'].shape[0],), device=device
            ).long()
            # forward process
            noise = torch.randn(naction.shape, device=device).float()
            noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
            sample_pred = model(noisy_actions, timesteps, 
                                obj_feat=nbatch_norm["obs"]["obj_feat"],
                                global_cond=nbatch_norm["obs"])   
            loss = nn.functional.mse_loss(sample_pred, naction_gt) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            loss_acc += loss.item()

        loss_acc /= len(dataloader)
        if use_wandb:
            wandb.log({"Train/Loss": loss_acc}, step=epoch_idx)
        else:
            writer.add_scalar('Loss/train', loss_acc, epoch_idx)
        epoch_end_time = datetime.datetime.now()
        print("Epoch %d, avg loss is %f" %(epoch_idx , loss_acc), flush=True)
        if epoch_idx < 5:
            print(" -------------- time for one epoch is %s ------" %(str(epoch_end_time - epoch_start_time)), flush=True)

        if epoch_idx % cfg["save_checkpt_epoch"] == 0 and epoch_idx > 10:
            model_util.save_model_optimizer_lrscheduler_checkpt(model, epoch_idx, optimizer, lr_scheduler, 
                                                                os.path.join(save_dir, "model_epoch_%d.pth" %(epoch_idx)))

    if use_wandb:
        wandb.run.finish()

    model_path = os.path.join(save_dir, "model_final.pth")
    model_util.save_model_optimizer_lrscheduler_checkpt(model, epoch_idx, optimizer, lr_scheduler, model_path)
    print("saving trained model at path: " + model_path, flush=True)
    return model

if __name__ == '__main__':
    start_overall = datetime.datetime.now()
    cfg = yaml_util.load_yaml(args.yaml_file)
    exp_name = cfg["exp_name"]
    save_dir = os.path.join(cfg["save_root_dir"], exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not cfg["use_wandb"]:
        writer = SummaryWriter(os.path.join(save_dir, "runs"))
    yaml_util.save_yaml(os.path.join(save_dir, "config.yaml"), cfg)
    if cfg["use_wandb"]:
        # Loggers
        wandb.init(config=cfg, project=cfg["wandb_pj_name"], entity=cfg["wandb_entity"],
                    name=exp_name, dir=save_dir)
        
    start_time = datetime.datetime.now()    
    print("starting time is: ")
    print(start_time)
    train_dataset = load_contact_dataset(cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], drop_last=True)      
    print("************************ In total, dataloading takes time: " + str(datetime.datetime.now() - start_time) + "************************")
    mesh_dict = train_dataset.mesh_dict
    stat_dict = data_util.load_stat_dict(cfg["stat_dict_path"], device)
    contact_model, optimizer, lr_scheduler = load_contact_module(cfg, device, train_dataloader)
    noise_scheduler = load_noise_scheduler(cfg)

    if cfg["num_epochs"] != 0:
        train(save_dir, train_dataloader, 
              noise_scheduler, contact_model, optimizer, lr_scheduler,
              cfg["start_epochs"], cfg["num_epochs"], use_wandb=cfg["use_wandb"])


    end_overall = datetime.datetime.now()
    print("Whole program execution time is: ")
    print(end_overall-start_overall)



