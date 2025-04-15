import wandb
import torch
import torch.nn as nn
import os
import copy
import matplotlib.pyplot as plt
import datetime
from utils import model_util, data_util
from utils import model_util
from utils import data_util
from torch.utils.tensorboard import SummaryWriter



class Trainer:

    def __init__(self, cfg, 
                 base_model, 
                 ema_model, 
                 optimizer, 
                 lr_scheduler, 
                 noise_scheduler,
                 train_dataloader,
                 stat_dict, 
                 device, 
                 save_dir):
        self.cfg = cfg
        self.base_model = base_model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.noise_scheduler = noise_scheduler
        self.dataloader = train_dataloader
        self.stat_dict = stat_dict
        self.device = device
        self.save_dir = save_dir
        self.use_wandb = self.cfg["train"]["use_wandb"]
        self.wandb_pj_name = self.cfg["train"]["wandb_pj_name"]
        self.entity = self.cfg["train"]["wandb_entity"]
        self.exp_name = self.cfg["exp_name"]
        if not self.use_wandb:
            self.writer = SummaryWriter(os.path.join(save_dir, "runs")) 


    def train(self):
        if self.use_wandb:
            wandb.init(config=self.cfg, project=self.wandb_pj_name, entity=self.entity,
                       name=self.exp_name, dir=self.save_dir)

        start_epochs = self.cfg["train"]["start_epochs"]
        num_epochs = self.cfg["train"]["num_epochs"] + start_epochs
        self.base_model.train()
        loss_list = []
        for epoch_idx in range(start_epochs, num_epochs):
            epoch_start = datetime.datetime.now()
            loss_acc = 0
            # batch loop
            for count, nbatch in enumerate(self.dataloader):                
                nbatch = data_util.preprocess_batch(nbatch, self.stat_dict, self.device)
                naction = nbatch['action'].to(self.device).float().clone()
                noise = torch.randn(naction.shape, device=self.device).float()
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (nbatch['action'].shape[0],), device=self.device
                ).long() # [bs]
        
                # forward process
                noisy_actions = self.noise_scheduler.add_noise(
                    original_samples=naction, noise=noise, timesteps=timesteps)

                # predict the sample
                diff_pred = self.base_model(noisy_actions, 
                                    timesteps, 
                                    obj_feat=nbatch["obs"]["object"], 
                                    contact_cond=nbatch["obs"]["contact_points"] if "contact_points" in nbatch["obs"] else None,
                                    global_states=nbatch["obs"]["global_states"] if "global_states" in nbatch["obs"] else None,
                                    contact_on_prob=self.cfg["train"]["contact_on_prob"]) 


                loss = nn.functional.mse_loss(diff_pred, naction)
                # optimize
                loss.backward()
                self.optimizer.step()     
                self.optimizer.zero_grad() 
                self.lr_scheduler.step()
                # update Exponential Moving Average of the model weights
                self.ema_model.step(self.base_model.parameters())
                loss_acc += loss

            loss_acc /= len(self.dataloader)
            loss_list.append(loss_acc.item())
            if self.use_wandb:
                wandb.log({"Train/Loss": loss_acc}, step=epoch_idx)
            # log loss
            else:
                self.writer.add_scalar('Loss/train', loss_acc, epoch_idx)
                for name, param in self.base_model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f'gradients/{name}', param.grad, epoch_idx)
        

            print("time taken for one epoch is: ")
            print(datetime.datetime.now() - epoch_start)

            # delete other prints for now
            print("Epoch %d, total loss is %f" %(epoch_idx , loss_acc), flush=True)
            if epoch_idx % self.cfg["train"]["save_checkpt_epoch"] == 0 and epoch_idx > 0:
                print("******************** saving model at epoch %d *******************" %epoch_idx, flush=True)
                ema_model_save = copy.deepcopy(self.base_model)
                self.ema_model.copy_to(ema_model_save.parameters())
                model_path = os.path.join(self.save_dir, "model_epoch_%d.pth" %epoch_idx)
                model_util.save_model_optimizer_lrscheduler_checkpt(ema_model_save, epoch_idx, 
                                                                    self.optimizer, self.lr_scheduler, 
                                                                    model_path, ema_model=self.ema_model)
                

        # Weights of the EMA model
        # is used for inference
        ema_model_save = copy.deepcopy(self.base_model)
        self.ema_model.copy_to(ema_model_save.parameters())
        model_util.save_model_optimizer_lrscheduler_checkpt(ema_model_save, epoch_idx, 
                                                            self.optimizer, self.lr_scheduler, 
                                                            os.path.join(self.save_dir, "model_final.pth"), 
                                                            ema_model=self.ema_model)
        print("saving trained model at path: " + (os.path.join(self.save_dir, "model_final.pth")), flush=True)
        plt.plot(loss_list)
        plt.savefig(os.path.join(self.save_dir, "loss_plot.png"))
        print('training complete')

        if self.use_wandb:
            wandb.run.finish()
        
        return ema_model_save
    
