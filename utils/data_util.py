
import numpy as np
import copy
import torch
from scipy.spatial.transform import Rotation as R
import pickle
import potpourri3d as pp3d
import trimesh
import os
import json 


def load_mesh_dict(base_dir="data/arctic_raw/", category_list=None):
    ''' load original raw object template from raw arctic data '''
    if category_list is None:
        category_list = ["box", "espressomachine", "ketchup", 
                        "laptop", "microwave", "mixer", "notebook", "phone", 
                        "scissors", "waffleiron", "capsulemachine"] # for onehot mask
    mesh_dict = {}
    for c in category_list:    
        meta_path = "meta/object_vtemplates/%s/" %c
        print("Loading object mesh template from " +  base_dir +  meta_path)
        obj_mesh = trimesh.load(os.path.join(base_dir, meta_path, "mesh.obj"),  process=False)
        with open(os.path.join(base_dir,meta_path,"parts.json")) as pf:
            parts = np.array(json.load(pf))
        mesh_dict[c] = {}
        mesh_dict[c]["verts"] = obj_mesh.vertices / 1000
        mesh_dict[c]["faces"] = obj_mesh.faces
        mesh_dict[c]["parts"] = parts
        mesh_dict[c]["scale"] = 1 
        mesh_dict[c]["centroid"] = np.zeros(3)
    return mesh_dict

def load_unit_mesh_dict(base_dir="assets/mesh_dict_unit_sphere_085_no_recenter.npy"):
    '''load normalized arctic object mesh dict'''
    return np.load(base_dir, allow_pickle=True).item()

def NpDictToTensor(a):
    '''only works for double nested dict
    '''
    b = copy.deepcopy(a)
    for k,v in b.items():
        if isinstance(v, dict):
            b[k] = NpDictToTensorSingle(v)
        else:
            b = NpDictToTensorSingle(b)
            break
    return b

def NpDictToTensorSingle(a):
    '''only works for single dict
    '''
    b = copy.deepcopy(a)
    for k,v in b.items():
        if k == "fitting_err":
            v = np.array(v)
            b[k] = torch.from_numpy(v).cuda().float()
        b[k] = torch.from_numpy(v).cuda().float()
    return b

def TensorDictToNp(aa):
    '''convert nested tensor dict to numpy array
    works for non-nested dict too
    '''
    if isinstance(aa, list):
        for a in aa:
            b = copy.deepcopy(a)
            for k,v in b.items():
                if isinstance(v, dict):
                    b[k] = TensorDictToNp(v)
                else:
                    if isinstance(v, torch.Tensor):
                        b[k] = v.detach().cpu().numpy()
    else:
        a = aa            
        b = copy.deepcopy(a)
        for k,v in b.items():
            if isinstance(v, dict):
                b[k] = TensorDictToNp(v)
            else:
                if isinstance(v, torch.Tensor):
                    b[k] = v.detach().cpu().numpy()
    return b


def object_verts_to_world(verts, parts, state):
    ''' 
    transform object vertices to world coordinate frame per frame
    Args:
        state: no array of [T, 7]: order: articulate -> rot -> trans
    return:
        the vertices of updated objects
    '''
    # step1 : transform top 
    verts_c = verts.copy()
    top = verts_c[parts == 0]
    top_index = np.where(parts == 0)[0]
    bottom = verts_c[parts == 1]
    bottom_index = np.where(parts == 1)[0]
    # transform top and bottom back to z-axis aligned canonical position
    R_art = R.from_rotvec((0, 0, -state[0])).as_matrix()
    top2 = top @ R_art.T
    # Step2: transform both top and bottom_obj
    Rg = R.from_rotvec((state[1], state[2], state[3])).as_matrix()
    verts2 = np.concatenate((top2, bottom))
    index2 = np.concatenate((top_index, bottom_index))
    verts2 = verts2 @ Rg.T
    # Step3: translate
    verts2 += state[-3:]
    # transform verts 2 back to normalized canonical position
    verts3 = np.zeros_like(verts2)
    # shuffle back
    verts3[index2] = verts2
    return verts3 


def object_verts_to_world_batch(verts, part_seg, state):
    '''
    return the updated object vertices in world coordinate frame
    args:
        verts: [N, 3]: 
        part_seg: [N, 1]
        state: [obs_horizon, 7], order: articulate -> rot -> trans:
    returns:
        [obs_horion, N, 3]
    '''
    verts_list = []
    for i in range(state.shape[0]):
        verts_list.append(object_verts_to_world(verts, part_seg, state[i]))
    return np.array(verts_list)


def canonicalize_pointcloud(pt, trans, rot):
    '''
    pt: bs*ph, kp, 3
    trans: bs*ph, 3
    rot: bs*ph, 3
    '''
    pt = pt - trans[:, None, :]
    rotmat = R.from_rotvec(rot).as_matrix()
    rotmat = np.linalg.inv(rotmat)
    rotated_pointcloud = np.einsum('bij,bpj->bpi', rotmat, pt)
    return rotated_pointcloud

def uncanonicalize_pointcloud(pt, trans, rot):
    '''
    pt: np array of ph, kp, 3
    trans: np array of ph, 3
    rot: np array of ph, 3

    Returns: np array of float32 of shape [ph, kp, 3]
    '''
    # change rot from axis angle to rotation matrix
    rotmat = R.from_rotvec(rot).as_matrix()
    test2 = np.einsum('bij,bpj->bpi', rotmat, pt) # (ph, kp, 3) (ph, 3, 3)
    test2 = test2 + trans[:, None, :]
    return test2.astype(np.float32)


def uncanonicalize_dirvec_batch(dirvec, rot):
    ''' 
    Args:
       dirvec: [bs, T, kp, 3]
       rot: rotation vector, shape [bs, T, 3]
    Returns:
       dirvec_world: [bs, T, kp, 3]
    '''
    # Compute rotation matrices from rotation vectors
    rotmat = R.from_rotvec(rot.reshape(-1, 3)).as_matrix()  # Now shape is [bs*T, 3, 3]
    rotmat = rotmat.reshape(rot.shape[0], rot.shape[1], 3, 3)  # Reshape to [bs, T, 3, 3]

    # Apply the rotation matrices to the direction vectors
    dirvec_world = np.einsum('btij,btnj->btni', rotmat, dirvec)

    return dirvec_world

def load_stat_dict(stat_dict_path, device):
    print("loading stat dict")
    with open(stat_dict_path, 'rb') as f:
        stat_dict = pickle.load(f)
        for key, sub_dict in stat_dict.items():
            print(key)
            if key == "n":
                continue
            else:
                for k, v in sub_dict.items():
                    print(k)
                    print(v.shape)
                    stat_dict[key][k] = torch.from_numpy(stat_dict[key][k]).to(device).float()
    return stat_dict


def normalize_contact_dict(data_dict, stat_dict):
    '''
    data_dict: consists of the following keys: ['onehot_mask', 'condition', 'aux', 'action'])
    action: [b, 1024]
    onehot_mask: [b, 11]
    '''
    data_dict = copy.deepcopy(data_dict)
    for key, value in data_dict.items():
        if key == "aux":
            continue    
        if key == "action":
            data_dict[key] = normalize_item(data_dict[key], stat_dict[key]["mean"], stat_dict[key]["std"])
        else:
            for k2, _ in value.items():
                if k2 == "category_onehot":
                    continue
                if "obj_feat" in k2 and k2 not in stat_dict.keys():
                    k2_old = "bps" # rename of variable
                    data_dict[key][k2] = normalize_item(data_dict[key][k2], stat_dict[k2_old]["mean"], stat_dict[k2_old]["std"])
                else:
                    data_dict[key][k2] = normalize_item(data_dict[key][k2], stat_dict[k2]["mean"], stat_dict[k2]["std"])
    return data_dict


def normalize_dict(data_dict, stat_dict, ignore_keys=["viz", "aux", "category_onehot"]):
    '''Assume the two dictionaries are on the same device
    '''
    data_dict = copy.deepcopy(data_dict) 
    for key, value in data_dict.items():
        if key in ignore_keys:
            continue
        if key == "obs":
            for key2, _ in value.items():
                if key2 in ignore_keys:
                    continue
                else:
                    data_dict[key][key2] = normalize_item(data_dict[key][key2], stat_dict[key2]["mean"], stat_dict[key2]["std"])
        elif key == "action":
            data_dict[key] = normalize_item(data_dict[key], stat_dict[key]["mean"], stat_dict[key]["std"])
        else:
            continue
    return data_dict


def unnormalize_dict(data_dict, stat_dict, ignore_keys=["viz", "aux", "category_onehot"]):
    data_dict = copy.deepcopy(data_dict)
    for key, value in data_dict.items():
        if key in ignore_keys:
            continue
        if key == "obs":
            for key2, _ in value.items():
                if key2 in ignore_keys:
                    continue
                else:
                    data_dict[key][key2] = unnormalize_item(data_dict[key][key2], stat_dict[key2]["mean"], stat_dict[key2]["std"])
        elif key == "action":
            data_dict[key] = unnormalize_item(data_dict[key], stat_dict[key]["mean"], stat_dict[key]["std"])
        else:
            continue
    return data_dict


def normalize_item(data, mean, std):
    '''
    data: [B, PH, D, (3)]
    mean: [D] or [D, 3]
    '''
    assert data.shape[-1] == mean.shape[-1]
    #assert len(mean.shape) == 1 -> not true for condition in contact_dict
    return (data - mean) / (std + 1e-10) #torch.clamp(std, min=1e-3)


def unnormalize_item(data, mean, std):
    '''
    data: [B, PH, D]
    mean: [D]
    '''
    if data.shape[-1] != mean.shape[-1]: # for obs
        data = data.reshape(data.shape[0], -1, mean.shape[-1])
    # assert len(mean.shape) == 1
    return data * (std + 1e-10) + mean

def slice_manodict(mano_dict, start, end, articulate=True):
    '''slice mano dict into a smaller dict
    '''
    seq_dict = {}
    seq_dict["left"] = {}
    seq_dict["right"] = {}
    for k, sub_dict in mano_dict.items():
        for k2, _ in sub_dict.items():
            if k2 == "shape":
                seq_dict[k][k2] = mano_dict[k][k2]
            elif k2 == "fitting_err":
                continue
            else:
                # this should be the only place where articulation needs to be considered in dataloading
                if k2 == "root":
                    # don't slice root
                    seq_dict[k][k2] = mano_dict[k][k2].copy()
                elif not articulate and (k2 == "pose" or k2 == "rot"):             
                    seq_dict[k][k2] = np.zeros_like(mano_dict[k][k2][start:end, :])
                else:
                    seq_dict[k][k2] = mano_dict[k][k2][start:end, :].copy()
                if start == end:
                    seq_dict[k][k2] = seq_dict[k][k2][None, :]
    return seq_dict


def get_bps_feature_from_tree(tree, bps_points):
    '''
    tree: cKD tree built for mesh vertices
    bps_points: M, 3
    '''
    mesh_verts = tree.data
    nearest_vertices = tree.query(bps_points)[1]
    selected_points = mesh_verts[nearest_vertices]
    bps_feat = selected_points - bps_points
    return bps_feat, nearest_vertices

def get_bps_feature_batch(trees, bps_points):
    '''
    mesh_verts: [B], build for each vertices
    bps_points: M, 3

    returns:
    bps_feat: B, M, 3
    nearest_vertices: B, M
    '''
    bps_feat_array = []
    nearest_vertices_array = []
    for i in range(len(trees)):
        nearest_vertices = trees[i].query(bps_points)[1]
        selected_points = trees[i].data[nearest_vertices]
        bps_feat = selected_points - bps_points
        bps_feat_array.append(bps_feat)
        nearest_vertices_array.append(nearest_vertices)
    return np.array(bps_feat_array), np.array(nearest_vertices_array)


def preprocess_batch(nbatch, stat_dict, device, ignore_keys=["aux", "viz"]):
    '''normalize, convert to float, move to device and flatten
    nbatch can contain a nested dictionary with "obs"
    '''
    for k in nbatch.keys():
        if k in ignore_keys:
            continue
        elif k == "obs":
            for key, _ in nbatch[k].items():
                nbatch[k][key] = nbatch[k][key].to(device).float()
        else:    
            nbatch[k] = nbatch[k].to(device).float()

    nbatch_norm = normalize_dict(nbatch, stat_dict, ignore_keys)
    return nbatch_norm


def preprocess_contact_batch(batch, device, stat_dict, normalize_data=True):
    for key, _ in batch.items():
        if key == "aux":
            continue
        if key == "obs":
            for k2 in batch[key]:
                batch[key][k2] = batch[key][k2].to(device).float()
        else:
            batch[key] = batch[key].to(device).float()
    if normalize_data:
        nbatch_norm = normalize_contact_dict(batch, stat_dict)
    else:
        nbatch_norm = copy.deepcopy(batch)
    return nbatch_norm


def densify_contact(obj_verts, bps_inds, sparse_contact):
    ''' this method converts contact scalar per BPS feature to the dense mesh
    Args: 
        obj_verts: (num_frames, num_verts, 3) vertices of the object
        bps_inds: (num_frames, num_bps) indices of the BPS features
        sparse_contact: (num_frames, num_bps) sparse contact map
    Returns:
        dense_contact_map: (num_frames, num_verts) dense contact map
    '''
    
    assert obj_verts.shape[0] == len(bps_inds)
    num_frames = len(obj_verts)
    dense_contact_list = []
    nan_exists = np.isnan(sparse_contact).any()
    assert not nan_exists, "nan exists in the sparse contact map, number of nan is {}".format(np.isnan(sparse_contact).sum())
    for i in range(num_frames):
        solver = pp3d.PointCloudHeatSolver(obj_verts[i])
        dense_contact_gt = solver.extend_scalar(bps_inds[i], sparse_contact[i])
        dense_contact_list.append(dense_contact_gt)
    return np.array(dense_contact_list)
