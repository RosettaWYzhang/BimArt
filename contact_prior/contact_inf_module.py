import numpy as np
import torch
from utils import data_util, yaml_util
from contact_prior.contact_prior_util import load_contact_module, load_noise_scheduler

class ContactInference():
    def __init__(self, mesh_dict, cfg, motion_config):
        '''Contact Inference Module.
        '''
        self.cfg = cfg
        self.motion_config = motion_config
        self.bps_points = np.load("assets/bps_normalized_part_based.npy")
        self.device = "cuda"
        self.noise_scheduler = load_noise_scheduler(self.cfg)
        self.load_model()
        self.mesh_dict = mesh_dict
        self.stat_dict = data_util.load_stat_dict(self.cfg["stat_dict_path"], "cuda")
        self.motion_stat_dict = data_util.load_stat_dict(self.motion_config["data"]["stat_dict_path"], "cuda")

    def load_model(self):
        self.model, _, _ = load_contact_module(self.cfg, self.device)
        self.model.eval()


    def prepare_oneshot_contact_data(self, motion_data):
        '''
        This function converts the motion data dictionary to contact data format.
        The unnormalization with motion stats and re-normalization with contact stats is to ensure consistency 
            in case the stats vary. 
        '''
        data = {}
        gt_contact = motion_data['obs']["contact_points"].clone()
        gt_contact = data_util.unnormalize_item(gt_contact, 
                                                self.motion_stat_dict["contact_points"]["mean"],
                                                self.motion_stat_dict["contact_points"]["std"])
        gt_contact = data_util.normalize_item(gt_contact, 
                                              self.stat_dict["action"]["mean"], 
                                              self.stat_dict["action"]["std"])
        data["action"] = gt_contact
        data["obs"] = {}
        obj_feat =  motion_data["obs"]["object"].clone()  
        obj_feat = data_util.unnormalize_item(obj_feat, 
                                              self.motion_stat_dict["object"]["mean"], 
                                              self.motion_stat_dict["object"]["std"])
        obj_feat = data_util.normalize_item(obj_feat, 
                                            self.stat_dict["obj_feat"]["mean"], 
                                            self.stat_dict["obj_feat"]["std"])
        global_states = motion_data["obs"]["global_states"].clone()
        global_states = data_util.unnormalize_item(global_states, 
                                                   self.motion_stat_dict["global_states"]["mean"], 
                                                   self.motion_stat_dict["global_states"]["std"])
        global_states = data_util.normalize_item(global_states, 
                                                 self.stat_dict["curr_global_states"]["mean"], 
                                                 self.stat_dict["curr_global_states"]["std"])
        data["obs"]["obj_feat"] = obj_feat 
        data["obs"]["curr_global_states"] = global_states
        data["aux"] = {}
        return data
    
    def infer_contacts(self, motion_data):
        '''
        Args:
            motion_data: a dictionary containing three top level keys: ['viz', 'action', 'obs']
            {"action": ground truth action for hand motion, not used in contact prediction
             "obs": ['object', 'global_states', 'contact_points']
             "viz: ['category', 'filename', 'left', 'right', 
                    'start_index', 'end_index', 'obj_world_state', 
                    'mano_dict', 'bps_viz_index', 'bps_feature', 
                    'gt_contact', 'action', 'gt_contact_points']
             }
        Returns: a tensor of shape (B, T, 2048) representing the predicted contact maps
        '''
        print("!!!!!!!!!!!!!!predict contact!!!!!!!!!!!!!!!!!")
        with torch.no_grad():
            data = self.prepare_oneshot_contact_data(motion_data)
            naction = torch.randn((data["action"].shape), device=self.device).float() #1, 
            self.noise_scheduler.set_timesteps(self.cfg["num_timesteps"])

            for k in self.noise_scheduler.timesteps: #[T..0]
                sample_pred = self.model(
                    sample=naction,
                    timestep=k,
                    obj_feat=data["obs"]["obj_feat"],
                    global_cond=data["obs"],
                )

                naction = self.noise_scheduler.step(
                    model_output=sample_pred, # x*
                    timestep=k,
                    sample=naction
                ).prev_sample # x_(t-1)
        return naction


    def postprocess_oneshot_contact(self, contact_pred, motion_data):
        '''This function adds predicted contact to the motion data dictionary before running motion model inference
        This predicted contact is unnormalized by contact stats and re-normalized by motion stats for consistency

        Args:
            contact_pred: a tensor of shape (B, T, 2048) representing the predicted contact maps
            motion_data: a dictionary containing three top level keys: ['viz', 'action', 'obs']
        '''
        contact_pred_unnorm = data_util.unnormalize_item(contact_pred.clone(), 
                                                         self.stat_dict["action"]["mean"], 
                                                         self.stat_dict["action"]["std"])
        contact_pred = data_util.normalize_item(contact_pred_unnorm, 
                                                self.motion_stat_dict["contact_points"]["mean"], 
                                                self.motion_stat_dict["contact_points"]["std"])
        motion_data["obs"]["contact_points"] = contact_pred
        return motion_data
   