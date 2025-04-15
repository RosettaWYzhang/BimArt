import numpy as np
import math
import torch
from utils import data_util,  mano_utils, viz_util
from contact_prior.contact_inf_module import ContactInference
import copy
import os
import datetime
import inference.guidance as guidance
from inference.postprocess import postprocess_output, visualize_output, get_mano_fit
from inference.optimization_refinement import PostOptimization


class InferenceModule: 
    def __init__(self, cfg, ema_model, dataloader, noise_scheduler, mesh_dict, stat_dict, 
                                    left_hand_index, right_hand_index,
                                    save_dir="",  prefix="", device="cuda", 
                                    contact_cfg=None, target_filename=None):
        self.cfg = cfg
        self.contact_cfg = contact_cfg  
        self.target_filename = target_filename
        self.actual_train_ph = cfg["data"]["pred_horizon"]    
        self.n_sample = cfg["test"]["n_sample"]
        self.start_sample = cfg["test"]["start_sample"]
        self.mesh_dict = mesh_dict 
        self.contact_model = ContactInference(mesh_dict, cfg=contact_cfg, motion_config=cfg)
        self.mano_layer = mano_utils.create_mano_layer()
        self.dataloader = dataloader
        self.ema_model = ema_model
        self.noise_scheduler = noise_scheduler
        self.stat_dict = stat_dict
        self.left_hand_index = left_hand_index
        self.right_hand_index = right_hand_index
        self.save_dir = save_dir
        self.prefix = prefix
        self.device = device
        self.post_optimizer = PostOptimization(save_dir=save_dir)


    def inference_with_contact_predict(self):
        eval_start_time = datetime.datetime.now()
        for b, test_batch in enumerate(self.dataloader):
            filename = test_batch["viz"]["filename"][-1]
            if self.target_filename is not None and filename != self.target_filename:
                continue
            print("Processing batch %d, filename: %s" %(b, filename))
            # save a copy for gt contact for visualization
            gt_contact_points = copy.deepcopy(test_batch["obs"]["contact_points"]) 
            test_batch["viz"]["gt_contact_points"] = gt_contact_points
            batch_eval_time = datetime.datetime.now()
            test_batch_norm = data_util.preprocess_batch(test_batch, self.stat_dict, device="cuda")
            # infer action
            with torch.no_grad():
                for i in range(self.start_sample, self.n_sample): 
                    print("running sample %d" %i)
                    sample_start_time = datetime.datetime.now()
                    pred_contact_list = self.contact_model.infer_contacts(test_batch_norm) 
                    test_batch_norm = self.contact_model.postprocess_oneshot_contact(
                        pred_contact_list, test_batch_norm)
                    # initialize action from Gaussian noise
                    actual_bs = test_batch_norm["action"].shape[0] if "action" in test_batch_norm else self.cfg["test"]["eval_batch_size"]
                    naction = torch.randn((actual_bs, self.actual_train_ph, self.cfg["data"]["action_dim"]), device=self.device).float()
                    # init scheduler
                    self.noise_scheduler.set_timesteps(self.cfg["train"]["num_diffusion_iters"])
                    mesh_viz_dict = self.mesh_dict[test_batch["viz"]["category"][0]]
                    # initialize guidance 
                    cost_guidance = guidance.Guidance(cfg=self.cfg, 
                                                      mano_layer=self.mano_layer, 
                                                      stat_dict=self.stat_dict, 
                                                      hand_index=self.left_hand_index,
                                                      test_batch=test_batch_norm, 
                                                      device="cuda",
                                                      w_cm=1,
                                                      guidance_scale=1,
                                                      mesh_viz_dict=mesh_viz_dict)
                    
            
                    for k in self.noise_scheduler.timesteps: #[T...0]
                        diff_pred = self.ema_model(naction, k, 
                                    obj_feat=test_batch_norm["obs"]["object"],
                                    contact_cond=test_batch_norm["obs"]["contact_points"],
                                    global_states=test_batch_norm["obs"]["global_states"],                                   
                                    contact_on_prob=1.0)   
                        
                        diff_pred = cost_guidance.step(diff_pred.detach())
                        diff_pred_uncond = self.ema_model(naction, k,  
                                                        obj_feat=test_batch_norm["obs"]["object"], 
                                                        global_states=test_batch_norm["obs"]["global_states"],
                                                        contact_on_prob=0)
                        diff_pred = (1+self.cfg["test"]["cfg_guide_strength"]) * diff_pred - self.cfg["test"]["cfg_guide_strength"] * diff_pred_uncond
          
    
                        naction = self.noise_scheduler.step(
                            model_output=diff_pred, # x*
                            timestep=k,
                            sample=naction
                        ).prev_sample # x_(t-1)

                    print("concatenated action shape is: ", naction.shape)
                    test_batch_unnorm = data_util.unnormalize_dict(test_batch_norm, self.stat_dict)    
                    naction = data_util.unnormalize_item(naction.detach().clone(), 
                                                        self.stat_dict["action"]["mean"], 
                                                        self.stat_dict["action"]["std"])
                    print("denormalized action shape is ", naction.shape)
       
                    out_dict = postprocess_output(test_batch_unnorm, naction, mesh_viz_dict, 
                                                 hand_keypoints=self.cfg["data"]["hand_keypoints"]) 
    
                    # Get an initial mano fit based on denoised data
                    mano_param_fit, left_verts, right_verts = get_mano_fit(
                        out_dict, self.mano_layer, 
                        idxl=self.left_hand_index, idxr=self.right_hand_index, 
                        ph=self.cfg["data"]["pred_horizon"], #include batch size here
                        mano_steps=self.cfg["viz"]["fit_mano_steps"], #if not use_cost_guidance else 100, 
                        hand_keypoints=self.cfg["data"]["hand_keypoints"]
                        ) 
                    
                    out_dict["pred"]["left"]["verts"] = left_verts
                    out_dict["pred"]["right"]["verts"] = right_verts
                    out_dict["mano_param"] = data_util.TensorDictToNp(mano_param_fit) 
                    
                    # optimization-refinement
                    out_dict = self.post_optimizer.refine_noisy_motions(out_dict)
                
                    if self.cfg["test"]["save_output"]:
                        save_output_dir = os.path.join(self.save_dir, "eval_output", self.prefix)
                        if not os.path.exists(save_output_dir):
                            os.makedirs(save_output_dir, exist_ok=True)
                            print("save_output, creating directory: " + save_output_dir)
                        
                        np.save(os.path.join(save_output_dir, "%s_batch%d_run%d_kp.npy" %(filename, b, i)), out_dict)
          
                        scene = visualize_output(out_dict, 
                                                 left_hand_faces=self.mano_layer["left"].faces, 
                                                 right_hand_faces=self.mano_layer["right"].faces,
                                                 norm_high=self.cfg["viz"]["norm_high"],)
                        #save scene as html
                        viz_save_dir = os.path.join(self.save_dir, "html", self.prefix)
                        if not os.path.exists(viz_save_dir):
                            # make dir recursively
                            os.makedirs(viz_save_dir, exist_ok=True)
                        scene.save_as_html(os.path.join(viz_save_dir, 
                                                        "%s_batch%d_run%d.html" %(filename, b, i)), 
                                           body_html=viz_util.get_legend(norm_high=self.cfg["viz"]["norm_high"]))     
                        print("saved scene at ", os.path.join(viz_save_dir, "%s_batch%d_run%d.html" %(filename, b, i)))   


                    sample_end_time = datetime.datetime.now()
                    print("--------------- Time taken for sample %d with batch size %d is -----------: " %(i, naction.shape[0]), 
                          sample_end_time - sample_start_time)       


            batch_eval_time = datetime.datetime.now() - batch_eval_time
            print("Time taken for batch %d is (include taking %d samples, save and visualize): " %(b, self.n_sample), batch_eval_time, flush=True)

        final_time = datetime.datetime.now()    
        print("finish!! Total time taken for evaluation: ", final_time - eval_start_time)


            
