
import smplx
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
import utils.data_util as data_util
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import trimesh
import pytorch3d



def create_mano_layer(use_flat_hand_mean=False):
    model_path = 'assets/mano_v1_2/'
    mano_scale = 1.0
    mano_layer = {
                'right': smplx.create(model_path, 'mano', use_pca=False, is_rhand=True, num_pca_comps=45, is_Euler=False, 
                                      flat_hand_mean=use_flat_hand_mean, scale=mano_scale).cuda(),
                'left': smplx.create(model_path, 'mano', use_pca=False, is_rhand=False, num_pca_comps=45, is_Euler=False, 
                                     flat_hand_mean=use_flat_hand_mean, scale=mano_scale).cuda()
                }
    return mano_layer

def get_hand_verts_joints(mano_layer, hand_type, mano_dict):
    '''
    Args:
    mano_dict: tensor array
    Returns: (as numpy array)    
        verts: (pred_horizon * bs, 778, 3)
        faces: (1538, 3)
        keypoints: [pred_horizon * bs, 16, 3] 16 is number of joints
    '''

    bs = mano_dict[hand_type]["rot"].shape[0]
    output = mano_layer[hand_type](global_orient=mano_dict[hand_type]["rot"],
                        hand_pose=mano_dict[hand_type]["pose"],
                        betas=torch.tile(mano_dict[hand_type]["shape"], (bs, 1)),
                        transl=mano_dict[hand_type]["trans"])

    verts = output.vertices.detach().cpu().numpy()
    output_joints = output.joints.detach().cpu().numpy()
    return verts, output_joints


def get_two_hand_verts_keypts_tensor(mano_layer, mano_dict):
    '''
    Args:
    mano_dict: a dictionary of mano parameters (tensors)
    Returns: tensors of left hand vertices, right hand vertices, left hand keypoints, right hand keypoints
    '''
    verts_l, joints_l = get_one_hand_verts_keypts_tensor(mano_layer, mano_dict, "left")
    verts_r, joints_r = get_one_hand_verts_keypts_tensor(mano_layer, mano_dict, "right")
    return verts_l, verts_r, joints_l, joints_r


def get_one_hand_verts_keypts_tensor(mano_layer, mano_dict, hand_type, canonicalise=False):
    '''
    Args:
    mano_dict: tensor dict
    Returns: single hand vertices and keypoints
    '''
    # check if it is numpy or tensor
    bs = mano_dict[hand_type]["rot"].shape[0]
    if canonicalise:
        mano_dict[hand_type]["rot"] = torch.zeros_like(mano_dict[hand_type]["rot"])
        mano_dict[hand_type]["trans"] = torch.zeros_like(mano_dict[hand_type]["trans"])
    
    output = mano_layer[hand_type](global_orient=mano_dict[hand_type]["rot"],
                        hand_pose=mano_dict[hand_type]["pose"],
                        betas=torch.tile(mano_dict[hand_type]["shape"], (bs, 1)),
                        transl=mano_dict[hand_type]["trans"])
    
    verts = output.vertices
    joints = output.joints
    return verts, joints

def transform_mano_parameters_tensordict(
    mano_layer,
    mano_dict,
    rigid_transformation,
):
    """Transforms SMPL-model family of parameters with a rigid transform.

    N always refers to the batch size.
    Args:
      smpl_layer: The SMPL-like module.
      root_rotation: The rotation of the root joint as a rotation matrix,
        (N, 3, 3).
      root_translation: The translation of the root, (N, 3).
      betas: The shape/identity vectors, (N, B).
      rigid_transformation: The rigid transformation to be applied, (N, 4, 4).
    Returns:
      A tuple containing the root rotation and translation that combine the
      previous root rotation and translation, and the desired rigid
      transformation.

    """
    rigid_transformation = torch.from_numpy(rigid_transformation).cuda().float()
    # Do a forward pass to get the joint positions for this identity.
    for s in ["left", "right"]:
        output = mano_layer[s](betas=mano_dict[s]["shape"][None, :])
        root_position = output.joints[:, 0]
        rotation_from_rigid = rigid_transformation[..., :3, :3]
        translation_from_rigid = rigid_transformation[..., :3, 3]
        new_root_rotation = torch.matmul(rotation_from_rigid, pytorch3d.transforms.axis_angle_to_matrix(mano_dict[s]["rot"].clone()))
        mano_dict[s]["rot"] = pytorch3d.transforms.matrix_to_axis_angle(new_root_rotation)
        
        new_translation = torch.einsum(
            '...mn,...n->...m',
            rotation_from_rigid,
            root_position + mano_dict[s]["trans"].clone()
        ) + translation_from_rigid - root_position
        mano_dict[s]["trans"] = new_translation
    return mano_dict

def create_homo_from_obj_trans_rot(obj_trans, obj_rot):
    '''
    Create a batch of inverse homogeneous transformation matrices.
    
    Parameters:
    obj_trans: N x 3 numpy array, each row is a translation vector.
    obj_rot: N x 3 numpy array, each row is a rotation vector in axis-angle format.

    Returns:
    N x 4 x 4 numpy array, where each slice along the first dimension is an inverse homogeneous transformation matrix.
    '''
    N = obj_trans.shape[0]
    # Create an array to hold the homogeneous matrices
    homo = np.tile(np.eye(4), (N, 1, 1))

    # Compute the rotation matrices from axis-angle representation
    obj_rotmat = R.from_rotvec(obj_rot).as_matrix()  # This already gives N x 3 x 3
    
    # Calculate the inverse rotation (transpose of rotation matrix for orthogonal matrices)
    inv_rot = np.transpose(obj_rotmat, (0, 2, 1))  # N x 3 x 3
    
    # Calculate the inverse translation
    inv_trans = -np.einsum("bij,bj->bi", inv_rot, obj_trans)  # N x 3
    
    # Set the rotation and translation in the homogeneous matrix
    homo[:, :3, :3] = inv_rot
    homo[:, :3, 3] = inv_trans

    return homo

def get_mano_offset(
    mano_layer,
    mano_dict,
    hand_type
):
    """return offset of the root joint for canonical pose
    Args:
      smpl_layer: The SMPL-like module.
    Returns:
      root positions Tensor
    """
    # convert mano_dict to tensor dict
    mano_dict_tensor = copy.deepcopy(mano_dict)
    # check if the dict consists of np array or tensor
    if isinstance(mano_dict_tensor[hand_type]["rot"], np.ndarray):
        mano_dict_tensor = data_util.NpDictToTensor(mano_dict_tensor)
    if len(mano_dict_tensor[hand_type]["shape"].shape) < 2:
        mano_dict_tensor[hand_type]["shape"] = mano_dict_tensor[hand_type]["shape"][None, :]
    output = mano_layer[hand_type](betas=mano_dict_tensor[hand_type]["shape"])
    root_position = output.joints[:, 0]
    return root_position


def initialize_mano_params(bs):
    '''initialize mano paramteers for optimization
    '''
    mano_param = {}
    mano_param["left"] = {}
    mano_param["right"] = {}
    mano_param["left"]["pose"] = torch.zeros((bs, 45), requires_grad=True, device="cuda")
    mano_param["left"]["rot"] = torch.zeros((bs, 3), requires_grad=True, device="cuda")
    mano_param["left"]["trans"] = torch.zeros((bs, 3), requires_grad=True, device="cuda")
    mano_param["right"]["pose"] = torch.zeros((bs, 45), requires_grad=True, device="cuda")
    mano_param["right"]["rot"] = torch.zeros((bs, 3), requires_grad=True, device="cuda")
    mano_param["right"]["trans"] = torch.zeros((bs, 3), requires_grad=True, device="cuda")
    mano_param["left"]["shape"] = torch.zeros((1, 10), requires_grad=True, device="cuda") 
    mano_param["right"]["shape"] = torch.zeros((1, 10), requires_grad=True, device="cuda")
    return mano_param


def make_mano_param_optimizable(mano_param_np):
    '''initialize mano paramteers for optimization from an existing numpy dicts
    '''
    mano_param = {}
    mano_param["left"] = {}
    mano_param["right"] = {}
    mano_param["left"]["pose"] = torch.from_numpy(mano_param_np["left"]["pose"]).cuda().requires_grad_(True).float()
    mano_param["left"]["rot"] = torch.from_numpy(mano_param_np["left"]["rot"]).cuda().requires_grad_(True).float()
    mano_param["left"]["trans"] = torch.from_numpy(mano_param_np["left"]["trans"]).cuda().requires_grad_(True).float()
    
    mano_param["right"]["pose"] = torch.from_numpy(mano_param_np["right"]["pose"]).cuda().requires_grad_(True).float()
    mano_param["right"]["rot"] = torch.from_numpy(mano_param_np["right"]["rot"]).cuda().requires_grad_(True).float()
    mano_param["right"]["trans"] = torch.from_numpy(mano_param_np["right"]["trans"]).cuda().requires_grad_(True).float()

    mano_param["left"]["shape"] = torch.from_numpy(mano_param_np["left"]["shape"]).cuda().requires_grad_(True).float()
    mano_param["right"]["shape"] = torch.from_numpy(mano_param_np["right"]["shape"]).cuda().requires_grad_(True).float()
    return mano_param

def root_align(mano_params, mano_layer, left_pred_kp, right_pred_kp, left_hand_index, right_hand_index, hand_keypoints=100):
    '''
    left_pred_kp: np array of B, N, 3

    returns:
    mano_params: tensor dict of mano parameters with updated root in type float
    '''
    mano_params = root_align_singlehand(mano_params, mano_layer, left_pred_kp, left_hand_index, hand_keypoints=hand_keypoints, hand_type="left")
    mano_params = root_align_singlehand(mano_params, mano_layer, right_pred_kp, right_hand_index, hand_keypoints=hand_keypoints, hand_type="right")
    return mano_params
    

def batch_affine_matrix_from_points(A, B):
    '''
    A, B: np array of B, N, 3

    returns: B, 4, 4 of type float
    '''
    # Initialize an empty list to store the resulting affine matrices
    affine_matrices = []
    
    # Iterate over each batch element
    for i, A_item in enumerate(A):
        # Compute affine matrix for the current batch element
        affine_matrix = trimesh.transformations.affine_matrix_from_points(A_item.T, B[i].T)
        
        # Append the affine matrix to the list
        affine_matrices.append(affine_matrix)
    
    # Convert the list of matrices to a numpy array
    return np.array(affine_matrices).astype(np.float32)


def root_align_singlehand(mano_params, mano_layer, pred_kp, hand_index, hand_keypoints=100, hand_type="left"):
    '''
    pred_kp: np array of BS, K, 3 (need to squeeze before pass in)
    '''

    iden_verts, iden_kp = get_one_hand_verts_keypts_tensor(mano_layer, mano_params, hand_type)
    if hand_index is not None:
        iden_verts = iden_verts.detach().cpu().numpy()[:, hand_index, :]
    else:
        iden_verts = iden_verts.detach().cpu().numpy()
    iden_kp = iden_kp.detach().cpu().numpy()
    if hand_keypoints == 100:
        M = batch_affine_matrix_from_points(iden_verts, pred_kp)
    else:
        M = batch_affine_matrix_from_points(iden_kp.squeeze(), pred_kp)
    Rot = M[:, :3, :3]
    Rot = R.from_matrix(Rot).as_rotvec().astype(np.float32)
    Trans = M[:, :3, 3]
    mano_params[hand_type]["rot"] = torch.from_numpy(Rot).cuda().requires_grad_(True)
    mano_params[hand_type]["trans"] = torch.from_numpy(Trans).cuda().requires_grad_(True)

    return mano_params


def fit_mano(joints_l_ref, joints_r_ref, mano_layer, ph, 
             idxl=None, idxr=None, hand_keypoints=100, steps=4000):
    
    ''' 
    joints_l_ref: BX100x3 tensor
    joints_r_ref: BX100x3 tensor
    idxl: index for left hand 
    idxr: index for right hand
    '''
    mano_params_list = []
    joints_l_list = []
    joints_r_list = []
    verts_l_list = []
    verts_r_list = [] 
    joints_l_sparse_list = []
    joints_r_sparse_list = []
    loss_mano_list = []
    for bs in range(joints_r_ref.shape[0]):
        print("Fitting mano for batch item %d" %bs, flush=True)
        with torch.enable_grad():
           
            mano_params = initialize_mano_params(ph)
            mano_params = root_align(mano_params, mano_layer, 
                                    joints_l_ref.detach().cpu().numpy()[bs], joints_r_ref.detach().cpu().numpy()[bs], 
                                    idxl, idxr, hand_keypoints=hand_keypoints)
        

            optimizer_mano = optim.Adam([mano_params['left']['pose'], 
                                    mano_params['left']['rot'], 
                                    mano_params['left']['trans'],
                                    mano_params['left']['shape'],
                                    mano_params['right']['pose'], 
                                    mano_params['right']['rot'], 
                                    mano_params['right']['trans'],
                                    mano_params['right']['shape']
                                    ], lr=0.001)
            # create loss function
            loss_func = nn.MSELoss()
            c = 0
            while c <= steps:
                # get keypoints from mano parameter
                verts_l, verts_r, joints_l_sparse, joints_r_sparse = get_two_hand_verts_keypts_tensor(mano_layer, mano_params)
                joints_l = verts_l[:, idxl]
                joints_r = verts_r[:, idxr]
                mse_loss = loss_func(joints_l, joints_l_ref[bs]) + loss_func(joints_r,  joints_r_ref[bs])
                if c == steps:
                    print("Mano Fit Loss", mse_loss.item(), flush=True)

                optimizer_mano.zero_grad()
                mse_loss.backward()
                optimizer_mano.step()
                c += 1
                if mse_loss < 1e-6:
                    break

            mano_params_list.append(mano_params)
            joints_l_list.append(joints_l.detach().cpu().numpy())
            joints_r_list.append(joints_r.detach().cpu().numpy())
            verts_l_list.append(verts_l.detach().cpu().numpy())
            verts_r_list.append(verts_r.detach().cpu().numpy())
            joints_l_sparse_list.append(joints_l_sparse.detach().cpu().numpy())
            joints_r_sparse_list.append(joints_r_sparse.detach().cpu().numpy())
            loss_mano_list.append(mse_loss.item())

    return (mano_params_list, np.array(verts_l_list), np.array(verts_r_list))

