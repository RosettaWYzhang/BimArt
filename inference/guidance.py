# define a set of objective functions used in guidance


import torch
from pytorch3d.ops import knn_points
import numpy as np
from utils import data_util

class Guidance:
    def __init__(self, cfg, 
                 mano_layer, 
                 stat_dict, 
                 hand_index,
                 test_batch, 
                 device="cuda",
                 w_cm=1,
                 guidance_scale=1,
                 mesh_viz_dict=None,
                 ):
        self.cfg = cfg
        self.device = device
        self.mano_layer = mano_layer
        self.ph = test_batch["action"].shape[1]
        self.w_cm = w_cm
        self.stat_dict = stat_dict  
        self.hand_index = hand_index
        self.test_batch = test_batch
        self.guidance_scale = guidance_scale
        self.mesh_viz_dict = mesh_viz_dict
        self.obj_init(test_batch)
        self.guidance_loss = {}
        self.guidance_loss["cm"] = []
        self.cm_init()
        
    def cm_init(self):
        self.cm_pred = self.test_batch["obs"]["contact_points"].clone().squeeze() # sparse
        self.cm_pred = data_util.unnormalize_item(self.cm_pred, 
                                self.stat_dict["contact_points"]["mean"], 
                                self.stat_dict["contact_points"]["std"])
        
    
    def get_left_and_right_hand_kp(self, naction_unnorm, hand_kp_num=100):
        '''slice hand keypoints from naction
        Args:
            naction_unnorm: [B, Ph, action_dim], assume B is always 1 here

        Returns:
            joints_l: [B, hand_kp_num, 3], left hand keypoints
            joints_r: [B, hand_kp_num, 3], right hand keypoints
        '''
        naction_joint = naction_unnorm.squeeze()
        naction_joint = naction_joint[:, 0:hand_kp_num*6]
        naction_joint = naction_joint.reshape((naction_joint.shape[0], -1, 3))
        joints_l = naction_joint[:, 0:hand_kp_num, :]
        joints_r = naction_joint[:, hand_kp_num:hand_kp_num*2, :]
        return joints_l, joints_r


    def get_obj_verts_cano_cuda(self, batch, mesh_viz_dict):
        '''return obj_verts and obj_faces as cuda tensor
        This function is called by cost guidance.

        Args:
            obj_verts_cuda: torch.Size([Ph, Nv, 3])
            obj_faces_cuda: torch.Size([Ph, Nf, 3])
        '''
        world_states = batch["viz"]["obj_world_state"].squeeze()
        obj_cano_state = np.zeros_like(world_states)
        obj_cano_state[:, 0] = world_states[:, 0]
        if "verts" in mesh_viz_dict:
            obj_verts = mesh_viz_dict["verts"].copy().squeeze()
        if "scale" in mesh_viz_dict:
            obj_verts = obj_verts / mesh_viz_dict["scale"]
        obj_verts = data_util.object_verts_to_world_batch(verts=obj_verts,
                                                        part_seg=mesh_viz_dict["parts"].copy().squeeze(), 
                                                        state=obj_cano_state)
        obj_faces = mesh_viz_dict["faces"].copy().squeeze()
        obj_verts_cuda = torch.from_numpy(obj_verts).cuda().requires_grad_(False).float()
        obj_faces_cuda = torch.from_numpy(obj_faces).cuda().requires_grad_(False)
        return obj_verts_cuda, obj_faces_cuda


    def obj_init(self, test_batch):
        obj_verts_cuda, obj_faces_cuda = self.get_obj_verts_cano_cuda(test_batch, self.mesh_viz_dict)
        self.obj_verts = obj_verts_cuda
        self.obj_faces = obj_faces_cuda
        self.bps_inds = self.test_batch["viz"]["bps_viz_index"].clone().squeeze().to(self.device)
        batch_indices = torch.arange(self.bps_inds.shape[0]).unsqueeze(1) 
        self.sparse_obj_verts = self.obj_verts[batch_indices, self.bps_inds]


    def step(self, diff_pred):                                      
        with torch.enable_grad():
            diff_pred.requires_grad_(True)
            diff_pred_unnorm = data_util.unnormalize_item(diff_pred,
                                    self.stat_dict["action"]["mean"], 
                                    self.stat_dict["action"]["std"])     

            left_hand_kp, right_hand_kp = self.get_left_and_right_hand_kp(diff_pred_unnorm)
            loss = 0
            l_cm = self.cm_discrepancy(left_hand_kp, self.cm_pred[:, :1024])
            l_cm += self.cm_discrepancy(right_hand_kp, self.cm_pred[:, 1024:]) 
            loss = loss + l_cm * self.w_cm
            gradients = torch.autograd.grad(outputs=loss,inputs=diff_pred,create_graph=False,retain_graph=False)[0]            
            # compute gradient norm
            grad_norm = torch.linalg.norm(gradients)    
            scale = 1.0 / (grad_norm + 1e-7) * self.guidance_scale
            diff_pred = diff_pred - scale * gradients 
        return diff_pred


    def minimum_euc_distance_from_A_to_B(self, A, B, k=1):
        """
        return: [frame_number, num_points_in_A]
        """
        # Step 1: Use knn_points to find the nearest point in A for each point in B
        knn = knn_points(A, B, K=k)  # Add batch dimension
        min_distances = torch.sqrt(knn.dists).squeeze(-1)   # Extract the distance to the nearest point in A (K=1)
        return min_distances


    def soft_nearest_neighbors(self, points_A, points_B, bandwidth=0.1):
        """
        Compute soft nearest neighbors from points_B to points_A using a Gaussian kernel.

        Args:
        - points_A: (N, D) tensor for N points in set A with D dimensions.
        - points_B: (M, D) tensor for M points in set B with D dimensions.
        - bandwidth: Bandwidth parameter for the Gaussian kernel.

        Returns:
        - distances: Soft nearest distances from points_B to points_A.
        """
        # Compute pairwise distances
        distances = torch.cdist(points_B, points_A)

        # Apply Gaussian kernel to weights
        weights = torch.exp(-distances / (2 * bandwidth**2))

        # Normalize weights
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Compute the weighted sum of distances
        soft_distances = (distances * weights).sum(dim=1)

        return soft_distances

    def cm_discrepancy(self, hand_vert, pred_cm):
        '''calculate the discrepancy between predicted contact maps and the derived contact maps from the predicted hand motions
        '''
        sparse_actual_cm = self.minimum_euc_distance_from_A_to_B(self.sparse_obj_verts, hand_vert)
        loss = torch.nn.functional.mse_loss(pred_cm, sparse_actual_cm)
        return loss
    
