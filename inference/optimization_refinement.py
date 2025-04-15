import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils import mano_utils
import torch
import torch.optim as optim
import pytorch3d.ops
from utils.contact_loss import compute_contact_loss


class PostOptimization():
    def __init__(self, save_dir):
        self.w_pen = 10
        self.w_proj = 100
        self.w_acc = 1000
        self.save_dir = save_dir
        self.op_steps = 100
        self.mano_layer = mano_utils.create_mano_layer()
        self.hand_index = np.load("assets/part_fps_hand_index_100.npy").astype(np.int32)


    def compute_proj_loss(self, proj_points, obj_verts):
        '''
        returns:
            loss: mean distance between projected points and object vertices
        '''
        dists, _,_ = pytorch3d.ops.knn_points(proj_points, obj_verts)
        loss = torch.mean(dists.squeeze())
        return loss
    
    @torch.enable_grad()
    def refine_noisy_motions(self, out_dict):
        '''
        This function optimizes the hand vertices using the penetration loss, contact loss and acceleration loss.

        Args:
            out_dict: dict, the output dictionary from the motion model before optimization.

        Returns:
            out_dict: the output dictionary after optimization.
        '''
        print("......Start optimization refinement..........")
        mano_param = out_dict["mano_param"]
        # convert mano param to optimizable tensors
        mano_params = mano_utils.make_mano_param_optimizable(mano_param)
        obj_verts = torch.from_numpy(out_dict["obj_verts"]).cuda().float().squeeze() # world
        obj_faces = torch.from_numpy(out_dict["obj_faces"]).cuda()

        # make joints_query optimizable
        right_dirvec = torch.from_numpy(out_dict["pred"]["right"]["dirvec"]).cuda().float().squeeze()
        left_dirvec = torch.from_numpy(out_dict["pred"]["left"]["dirvec"]).cuda().float().squeeze()
        joints_dirvec = torch.cat([left_dirvec, right_dirvec], dim=1).clone()
    
        optimizer = optim.AdamW([mano_params['left']['pose'],
                                mano_params['left']['rot'],
                                mano_params['left']['trans'],
                                mano_params['left']['shape'],
                                mano_params['right']['pose'],
                                mano_params['right']['rot'],
                                mano_params['right']['trans'],
                                mano_params['right']['shape']
                                ], lr=5e-4)

        for i in range(self.op_steps):
            optimizer.zero_grad()
            left_hand_vertices, right_hand_vertices, _, _ = mano_utils.get_two_hand_verts_keypts_tensor(self.mano_layer, mano_params)
            joints_query = torch.cat([left_hand_vertices, right_hand_vertices], dim=1)
            keypoints_query = torch.cat([left_hand_vertices[:, self.hand_index], right_hand_vertices[:, self.hand_index]], dim=1)
            joints_projection = joints_dirvec + keypoints_query
            # split into batches to save memory
            inter = 2 
            bs =  keypoints_query.shape[0] // inter
            pen_loss = torch.tensor(0.0).cuda()
            for f in range(bs):
                _, pen_loss_frame, _, _ = compute_contact_loss(
                    joints_query[f*inter:(f+1)*inter], obj_verts[f*inter:(f+1)*inter], obj_faces)
                pen_loss = pen_loss + pen_loss_frame
            if pen_loss > 0:
                pen_loss = pen_loss * self.w_pen
            else:
                pen_loss = torch.tensor(0.0).cuda()
               
            proj_loss  = self.compute_proj_loss(joints_projection, obj_verts)
            proj_loss = proj_loss * self.w_proj
            joint_smooth2 = torch.mean(torch.square(joints_query[:-2] - 2 * joints_query[1:-1] + joints_query[2:])) * self.w_acc
            loss = pen_loss + joint_smooth2 + proj_loss
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print(f"step {i},  pen loss: {pen_loss.item()}",
                    f"joint smooth2: {joint_smooth2.item()}",
                    f"proj loss: {proj_loss.item()}", f"total loss: {loss.item()}")


        print("Done with optimization refinement")
        # update the original out_dict with the optimized vertices
        out_dict["pred"]["left"]["verts"] = left_hand_vertices.detach().cpu().numpy()
        out_dict["pred"]["right"]["verts"] = right_hand_vertices.detach().cpu().numpy()
        out_dict['pred']['left']['joints'] = left_hand_vertices.detach().cpu().numpy()[:, self.hand_index, :] 
        out_dict['pred']['right']['joints'] = right_hand_vertices.detach().cpu().numpy()[:, self.hand_index, :]
        out_dict["mano_param"] = mano_params
        return out_dict

