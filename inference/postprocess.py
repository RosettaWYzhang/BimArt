import numpy as np
import torch
import utils.mano_utils as mano_utils
import utils.viz_util as viz_util
import utils.data_util as data_util


def postprocess_obj(mesh_dict, gt_obj_state):
    '''
    post-process object vertices to world space
    Args:
        gt_obj_state: [B, Ph, 7]
    Returns:
        obj_verts_world: [B, Ph, V, 3]
    '''
    obj_verts = mesh_dict["verts"].copy()
    if "scale" in mesh_dict:
        obj_verts = obj_verts / mesh_dict["scale"] + mesh_dict["centroid"]
    # here, we assume it is the same category
    gt_obj_state_reshape = gt_obj_state.reshape(-1, 7)
    obj_verts_world = data_util.object_verts_to_world_batch(obj_verts.squeeze(),
                                                            mesh_dict["parts"].squeeze(), 
                                                            gt_obj_state_reshape)
    obj_verts_world = obj_verts_world.reshape(gt_obj_state.shape[0], gt_obj_state.shape[1], -1, 3) # bs, T, V, 
    return obj_verts_world

def postprocess_dirvec(action_pred, hand_keypoints=100):
    '''only return dirvec in canonical space
    Args: action_pred: [B, Ph, action_dim]
    don't collapse the batch dimension
    '''    
     # B, 200, 1200
    dirvec_rel = action_pred[:, :, hand_keypoints*6:hand_keypoints*12].reshape(
        (action_pred.shape[0], action_pred.shape[1], -1, 3)) # exclude dirvec
    dirvec_left = dirvec_rel[:, :, 0:hand_keypoints, :]
    dirvec_right = dirvec_rel[:, :, hand_keypoints:hand_keypoints*2, :]
    return dirvec_left, dirvec_right


def postprocess_hand_sampled_points(action_pred, trans_base, rot_base, hand_keypoints=100):
    ''' slice the predicted surface points from action, return both the canonical and world space
    naction: numpy array [B, Ph, action_dim]
    trans_base: [B, Ph, 3]
    rot_base: [B, Ph, 3]
    left_wrist: [ph, 3]
    '''
    # process hand prediction: for all batch
    # collapse the batch dimension for uncanonicalization
    joints_pred_rel = action_pred[:, :, 0:hand_keypoints*6].reshape(
        (action_pred.shape[0] * action_pred.shape[1], -1, 3))
    joints_pred_world = data_util.uncanonicalize_pointcloud(
        joints_pred_rel, trans_base.reshape(-1, 3), rot_base.reshape(-1, 3))
    joints_pred_world = joints_pred_world.reshape(
        action_pred.shape[0],  action_pred.shape[1], hand_keypoints*6) # N, K, 3
    # N, 3
    # reshape to be consistent to above [32 * 16, 21, 3]
    joints_l = joints_pred_world[:, :, 0:hand_keypoints*3].reshape(-1, hand_keypoints, 3)
    joints_r = joints_pred_world[:, :, hand_keypoints*3:hand_keypoints*2*3].reshape(-1, hand_keypoints, 3)
    # reshape to maintain the batch dimension
    joints_l = joints_l.reshape(action_pred.shape[0], action_pred.shape[1], hand_keypoints, 3)
    joints_r = joints_r.reshape(action_pred.shape[0], action_pred.shape[1], hand_keypoints, 3)
    return joints_l, joints_r


def postprocess_gt_hand_verts(test_batch):
    verts_gt_l = test_batch["viz"]["left"]["hand_verts"].reshape(-1, 778, 3)
    verts_gt_r = test_batch["viz"]["right"]["hand_verts"].reshape(-1, 778, 3)
    # reshape to maintain the batch dimension
    verts_gt_l = verts_gt_l.reshape(test_batch["viz"]["left"]["hand_verts"].shape[0], 
                                    test_batch["viz"]["left"]["hand_verts"].shape[1], 778, 3)
    verts_gt_r = verts_gt_r.reshape(test_batch["viz"]["right"]["hand_verts"].shape[0], 
                                    test_batch["viz"]["right"]["hand_verts"].shape[1], 778, 3)
    return verts_gt_l, verts_gt_r


def postprocess_twohand_contact(obj_verts_world, bps_inds_sparse, pred_array):
    '''
    Since the contact value is predicted per BPS,
    this function densifies the contact value to object vertices per frame.

    Args: 
        obj_verts_world: bs, T, V, 3
        bps_inds_sparse: bs, T, 1024
        pred_array: bs, T, 2048
    Returns:
        combined_contact_list: bs, T, V
        dense_left_verts_contact_list: bs, T, V
        dense_right_verts_contact_list: bs, T, V
    '''
    combined_contact_list = []
    dense_left_verts_contact_list = []
    dense_right_verts_contact_list = []
    for bs in range(pred_array.shape[0]):
        pred_left_array = pred_array[bs, :, :1024]
        pred_right_array = pred_array[bs, :, 1024:]
        dense_left_verts_contact = data_util.densify_contact(obj_verts_world[bs], bps_inds_sparse[bs], pred_left_array)
        dense_right_verts_contact = data_util.densify_contact(obj_verts_world[bs], bps_inds_sparse[bs], pred_right_array)
        combined = np.minimum(dense_left_verts_contact, dense_right_verts_contact)
        combined_contact_list.append(combined)
        dense_left_verts_contact_list.append(dense_left_verts_contact)
        dense_right_verts_contact_list.append(dense_right_verts_contact)
    return np.array(combined_contact_list), np.array(dense_left_verts_contact_list), np.array(dense_right_verts_contact_list)


def add_world_dirvec_to_output(out_dict):
    '''process dirvec to get to world space and add to out_dict'''
    out_dict["pred"]["left"]["dirvec"] = data_util.uncanonicalize_dirvec_batch(
    out_dict["pred"]["left"]["dirvec_rel"], out_dict["obj_gt_states"][:, :, 1:4])
    out_dict["pred"]["right"]["dirvec"] = data_util.uncanonicalize_dirvec_batch(
        out_dict["pred"]["right"]["dirvec_rel"], out_dict["obj_gt_states"][:, :, 1:4])
    return out_dict


def postprocess_output(test_batch, naction, mesh_dict, hand_keypoints):
    '''post-process outputs for visualization and metric computation

    Args: 
        test_batch: unnormalized batch data
        naction: denoised action, unnormalized

    Return: a dictionary of items for visualization and metric computation
    '''
    naction = naction.detach().clone().cpu().numpy()
    gt_obj_state = test_batch["viz"]["obj_world_state"].detach().cpu().numpy().copy()
    trans_base = gt_obj_state[:, :, 4:7] # 43, 16, 3
    rot_base = gt_obj_state[:, :, 1:4]
    obj_verts = postprocess_obj(mesh_dict, gt_obj_state)

    joints_l, joints_r = postprocess_hand_sampled_points(
        naction, trans_base, rot_base, hand_keypoints)
    
    cano_dirvec_left, cano_dirvec_right = postprocess_dirvec(naction, hand_keypoints)
    verts_gt_l, verts_gt_r = postprocess_gt_hand_verts(test_batch)

    out_dict = {}
    out_dict["pred"] = {}
    out_dict["gt"] = {}
    out_dict["pred"]["left"] = {}
    out_dict["pred"]["right"] = {}
    out_dict["gt"]["left"] = {}
    out_dict["gt"]["right"] = {}

    out_dict["pred"]["left"]["joints"] = joints_l
    out_dict["pred"]["right"]["joints"] = joints_r
    out_dict["pred"]["left"]["dirvec_rel"] = cano_dirvec_left
    out_dict["pred"]["right"]["dirvec_rel"] = cano_dirvec_right
    out_dict["obj_gt_states"] = gt_obj_state.copy() 
    out_dict = add_world_dirvec_to_output(out_dict)
    out_dict["gt"]["left"]["verts"] = verts_gt_l.detach().cpu().numpy().copy()
    out_dict["gt"]["right"]["verts"] = verts_gt_r.detach().cpu().numpy().copy()
    out_dict["obj_verts"] = obj_verts
    out_dict["obj_faces"] = mesh_dict["faces"].copy().squeeze()
    out_dict["obj_parts"] = mesh_dict["parts"].copy().squeeze()
    contact_values = test_batch["obs"]["contact_points"].detach().cpu().numpy().copy()
    out_dict["obj_sparse_contact"] = contact_values
    contact_pred, left_contact, right_contact = postprocess_twohand_contact(obj_verts, 
                                                test_batch["viz"]["bps_viz_index"].detach().cpu().numpy().copy(),
                                                out_dict["obj_sparse_contact"])
    out_dict["combined_contact"] = contact_pred
    out_dict["separate_contact_left"] = left_contact
    out_dict["separate_contact_right"] = right_contact

    return out_dict


def get_mano_fit(out_dict, mano_layer, idxl=None, idxr=None, ph=64, mano_steps=100, hand_keypoints=100):
    left_hand_kp = out_dict["pred"]["left"]["joints"]
    right_hand_kp = out_dict["pred"]["right"]["joints"]
    joints_l_ref = torch.from_numpy(left_hand_kp).cuda()
    joints_r_ref = torch.from_numpy(right_hand_kp).cuda()

    fitted_mano_param, left_hand_vertices, right_hand_vertices  = mano_utils.fit_mano(
        joints_l_ref, joints_r_ref, mano_layer, ph,
        idxl=idxl, idxr=idxr, hand_keypoints=hand_keypoints,
        steps=mano_steps)

    return fitted_mano_param, left_hand_vertices, right_hand_vertices


def visualize_output(out_dict, left_hand_faces, right_hand_faces, norm_high):
    '''
    Visualize the generated motion using the Scenepic library
    Args:
        out_dict: a dictionary containing the output of the model
        norm_high: the maximum value for contact color normalization and colorbar legend
    Returns:
        scene: a Scenepic scene object which is subsequently saved as html
    '''   
    # assume we only visualize the first example in a batch
    viz_idx = 0
    scene = viz_util.build_scenepic_motion(obj_faces=out_dict["obj_faces"], 
                                           obj_verts_gt=out_dict["obj_verts"][viz_idx], 
                                           obj_parts=out_dict["obj_parts"], 
                                           obj_gt_states=out_dict["obj_gt_states"][viz_idx], 
                                           left_hand_vertices=out_dict["pred"]["left"]["verts"], 
                                           right_hand_vertices=out_dict["pred"]["right"]["verts"],
                                           left_hand_faces=left_hand_faces,
                                           right_hand_faces=right_hand_faces,                                                                       
                                           contact_scalar=out_dict["combined_contact"][viz_idx],
                                           pred_left_contact = out_dict["separate_contact_left"][viz_idx],
                                           pred_right_contact = out_dict["separate_contact_right"][viz_idx],
                                           norm_high=norm_high)
    print("......Finish building scene..........")
    return scene


