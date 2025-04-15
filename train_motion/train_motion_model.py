import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import sys
import argparse
import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import utils.data_util as data_util
import utils.yaml_util as yaml_util
import utils.model_util as model_util
from train_motion.bimart_trainer import Trainer
from dataset import motion_data

torch.set_printoptions(precision=10)
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--yaml_file', type=str, default="config_files/train_motion_config.yaml")
args = parser.parse_args()
print(args)


def main():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # **************************************** Data Loading *****************************************
    print("Experiment folder is created at: " + save_dir, flush=True)
    stat_dict = data_util.load_stat_dict(cfg["data"]["stat_dict_path"], device)    
    load_start = datetime.datetime.now()  
    train_dataset = motion_data.MotionDataset(cfg, split="train", return_aux_info=False)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["train"]["train_batch_size"],
                                  shuffle=True, num_workers=cfg["data"]["num_workers"], 
                                  pin_memory=True)
    load_end = datetime.datetime.now()
    print("FINISHED LOADING DATA, time taken is :")
    print(load_end - load_start)               
 
    # **************************************** Model Related *****************************************
    base_model = model_util.createNN(cfg, device)
    optimizer = model_util.create_optimizer(base_model, lr=cfg["train"]["lr"])
    ema_model = model_util.create_ema_model(base_model)
    noise_scheduler = model_util.createScheduler(cfg)
    lr_scheduler = model_util.create_lr_scheduler(optimizer, 
                                                  cfg["train"]["warmup_steps"], 
                                                  len(train_dataset), 
                                                  cfg["train"]["num_epochs"])

    if cfg["train"]["load_pretrained"]:
        model_file = os.path.join(save_dir, cfg["train"]["pretrain_model_name"])
        print("trying to load model from model file: ", model_file)
        if os.path.exists(model_file):
            print("--- loading pretrained model from --- " + model_file, flush=True)
            base_model, ema_model, _, _, epoch = model_util.load_model_optimizer_lrscheduler_checkpt(
                base_model, optimizer=None, lr_scheduler=None, model_path=model_file, ema_model=ema_model)
            cfg["train"]["start_epochs"] = epoch
            print("loaded model at epoch: ", epoch)
        else:
            print("pretrained model not found, retraining a new model")
    

    start = datetime.datetime.now()
    # initialize trainer
    trainer = Trainer(cfg=cfg,
                        base_model=base_model,
                        ema_model=ema_model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        noise_scheduler=noise_scheduler,
                        train_dataloader=train_dataloader,
                        stat_dict=stat_dict,
                        device=device,
                        save_dir=save_dir)

    trainer.train()
    print("training finished!")
    # check if diffusion model produce the same results by removing variance
    end = datetime.datetime.now()
    print("training time is: ")
    print(end-start)




if __name__ == "__main__":
    start_overall = datetime.datetime.now()
    torch.set_default_dtype(torch.float32)
    cfg = yaml_util.load_yaml(args.yaml_file)
    save_dir = os.path.join(cfg["save_root_dir"], cfg["exp_name"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    yaml_util.save_yaml(os.path.join(save_dir, "config.yaml"), cfg)
    device = torch.device('cuda')
    main()
    end_overall = datetime.datetime.now()
    print("Whole program execution time is: ")
    print(end_overall-start_overall)




