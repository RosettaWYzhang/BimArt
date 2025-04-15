import numpy as np
import sys
import os
from torch.utils.data import DataLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils import data_util, model_util, yaml_util
import datetime
import argparse
from dataset import motion_data
from inference.inference_module import InferenceModule

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default="config_files/bimart_inference.yaml")
parser.add_argument('--target_filename', type=str, default="", 
                    help="target test sequence to eval, either select a line from assets/test_filenames or leave it empty")
args = parser.parse_args()
print(args)


if __name__ == "__main__":
    # add config file as an argument
    config_file = args.config_file
    cfg = yaml_util.load_yaml(config_file)
    if args.target_filename != "":
        print("************ target file name is %s ********************" % args.target_filename)
        category = args.target_filename.split("_")[0]
        print("Target filename overwrite category argument: category is set to: ", category)

    save_dir = os.path.join(cfg["save_root_dir"], cfg["exp_name"])
    device = "cuda"
    start_time = datetime.datetime.now()
    contact_config_file = cfg["test"]["contact_config_file"]
    contact_cfg = yaml_util.load_yaml(contact_config_file)
    print("Contact Exp Name is : ", contact_cfg["exp_name"])
    # **************************************** Data Related *****************************************
    print("Experiment folder is created at: " + save_dir, flush=True)
    stat_dict = data_util.load_stat_dict(cfg["data"]["stat_dict_path"], device)

    # Get test data
    test_dataset = motion_data.MotionDataset(cfg, split="test", return_aux_info=True)    
    mesh_dict = test_dataset.mesh_dict                          
    test_dataloader = DataLoader(test_dataset, batch_size=cfg["test"]["eval_batch_size"],
                shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

            
    # **************************************** Model Related *****************************************
    base_model = model_util.createNN(cfg,  device)
    optimizer = None
    ema_model = model_util.create_ema_model(base_model)
    noise_scheduler = model_util.createScheduler(cfg)
    lr_scheduler = None
  
    model_file = os.path.join(save_dir, cfg["train"]["pretrain_model_name"])
    print("searching for saved model file: ", model_file)
    if os.path.exists(model_file):
        print("--- loading pretrained model from --- " + model_file, flush=True)
        ema_model, _, optimizer, lr_scheduler, epoch = model_util.load_model_optimizer_lrscheduler_checkpt(
            base_model, optimizer, lr_scheduler, model_file, ema_model=None)     
        cfg["train"]["start_epochs"] = epoch
        print("loaded model at epoch: ", epoch)
    else:
        print("motion pretrained model not found, exiting")
        exit(0)
   
    ema_model.eval()
    left_hand_index = np.load(cfg["data"]["hand_index_path"]).astype(np.int32)
    right_hand_index = left_hand_index
    start = datetime.datetime.now()
    # generate diverse results
    sampler = InferenceModule(cfg, ema_model=ema_model, dataloader=test_dataloader, 
                        noise_scheduler=noise_scheduler, mesh_dict=mesh_dict, stat_dict=stat_dict, 
                        left_hand_index=left_hand_index, right_hand_index=right_hand_index,
                        save_dir=save_dir, 
                        prefix="test_%s"%cfg["viz"]["viz_prefix"], device=device,
                        contact_cfg=contact_cfg,
                        target_filename=args.target_filename if args.target_filename != "" else None) # return [1, 16, 102]
    sampler.inference_with_contact_predict()
    print("Total time taken for evaluation is: ", datetime.datetime.now() - start)

