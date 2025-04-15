'''
This script process the hand features for the BimArt dataset from the raw arctic data sequences.
'''
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import numpy as np
import glob
import copy
import utils.mano_utils as mano_utils
import utils.data_util as data_util
from scipy.spatial import cKDTree
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_base_dir', default="data/arctic_raw",
                    type=str,
                    help='path of the downloaded artic data')
parser.add_argument('--data_process_dir',
                    default="data/arctic_processed_data",
                    type=str, help='path of the processed artic data')
args = parser.parse_args()


class HandFeaturePreprocess():
    def __init__(self):
        '''
        The class extracts hand features.
        '''
        self.file_suffix = "processed_hand_features"
        self.base_dir = args.data_base_dir
        self.hand_index_path = "assets/part_fps_hand_index_100.npy"
        self.load_obj_mano_files()
        self.obj_files = self.data_files["obj_files"]
        self.mano_files = self.data_files["mano_files"]
        self.obj_t = 4
        self.left_hand_index = np.load(self.hand_index_path).astype(np.int32)
        self.right_hand_index = self.left_hand_index
        # subsample mesh vertices
        self.mano_layer = mano_utils.create_mano_layer()
        # obj state index for translation
        self.__process_raw_data__()



    def process_hand(self, file_cat, seq_len, mano_dict, obj_state_div):
        obj_state_artionly = obj_state_div.copy() 
        obj_state_artionly[:, 1:] = np.zeros_like(obj_state_div[:, 1:])
        obj_state_arti_rot = np.zeros_like(obj_state_div)
        obj_state_arti_rot[:, 0:self.obj_t] = obj_state_div[:, 0:self.obj_t]
        # dense
        obj_cano_verts_dense = data_util.object_verts_to_world_batch(self.mesh_dict[file_cat]["verts"], 
                                                                     self.mesh_dict[file_cat]["parts"], 
                                                                     obj_state_artionly)

        # do mano forward pass before transformation
        mano_tensor_dict = data_util.NpDictToTensor(mano_dict)
        hand_verts_l_all, keypoints_l_all = mano_utils.get_hand_verts_joints(
            self.mano_layer, "left", mano_tensor_dict)
        hand_verts_r_all, keypoints_r_all = mano_utils.get_hand_verts_joints(
            self.mano_layer, "right", mano_tensor_dict)
        left_root = mano_utils.get_mano_offset(self.mano_layer, mano_dict,"left").detach().cpu().numpy()
        right_root = mano_utils.get_mano_offset(self.mano_layer, mano_dict,"right").detach().cpu().numpy()
        mano_dict["left"]["root"] = left_root
        mano_dict["right"]["root"] = right_root

        # get mano parameters in canonical space and concatenate them
        mano_dict_cano = copy.deepcopy(mano_tensor_dict)
        obj_rot = obj_state_div[:, 1:self.obj_t]
        obj_trans = obj_state_div[:, self.obj_t:]
        rigid_trans_inv = mano_utils.create_homo_from_obj_trans_rot(obj_trans, obj_rot)
        mano_dict_cano = mano_utils.transform_mano_parameters_tensordict(self.mano_layer, mano_dict_cano, rigid_trans_inv)

        # keypoinst_l dim: pred_dim, 16, 3
        hand_sampled_l = hand_verts_l_all[:, self.left_hand_index]
        hand_sampled_r = hand_verts_r_all[:, self.right_hand_index]
        hand_sampled_l_cano = data_util.canonicalize_pointcloud(
            hand_sampled_l, obj_state_div[:, self.obj_t:], obj_state_div[:, 1:self.obj_t])
        hand_sampled_r_cano = data_util.canonicalize_pointcloud(
            hand_sampled_r, obj_state_div[:, self.obj_t:], obj_state_div[:, 1:self.obj_t])
        processed_data = {}
        processed_data["category"] = file_cat
        processed_data["mano_dict"] = mano_dict
        # hand related 
        processed_data["left_hand_verts"] = hand_verts_l_all
        processed_data["right_hand_verts"] = hand_verts_r_all
        processed_data["left_hand_sampled_verts"] = hand_sampled_l
        processed_data["right_hand_sampled_verts"] = hand_sampled_r
        processed_data["right_hand_sampled_verts_cano"] = hand_sampled_r_cano
        processed_data["left_hand_sampled_verts_cano"] = hand_sampled_l_cano
        processed_data["left_hand_keypoints"] = keypoints_l_all
        processed_data["right_hand_keypoints"] = keypoints_r_all
        processed_data["kp"] = self.get_action_kp_feature(0, seq_len, processed_data)
        processed_data["dirvec"] = self.get_dirvec_feature(0, seq_len, processed_data, obj_cano_verts_dense)
        return processed_data



    def __process_raw_data__(self):
        print("loading %d files" %len(self.obj_files))
        for f in range(len(self.obj_files)):
            file_cat = self.obj_files[f].split("/")[-1].split("_")[0]
            seq_name = self.obj_files[f].split("/")[-2]
            filename = self.obj_files[f].split("/")[-1].split(".")[0] + "_%s.npy"%self.file_suffix 
            print("process category %s" %file_cat)
            save_dir = os.path.join(args.data_process_dir, file_cat, seq_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print("............load file %d...................." %f, flush=True) 
            obj_state_div = np.load(self.obj_files[f], allow_pickle=True)
            seq_len = obj_state_div.shape[0]
            mano_dict = np.load(self.mano_files[f], allow_pickle=True).item()
            obj_state_div[:, self.obj_t:] /= 1000 # divide translation, so that it is consistency with hand unit
            processed_data = self.process_hand(file_cat, seq_len, mano_dict, obj_state_div) 
            np.save(os.path.join(save_dir,filename), processed_data)
            print("saved file %s" %(save_dir + filename))


    def get_action_kp_feature(self, base_index, goal_index, process_data):
        sampled_l = process_data["left_hand_sampled_verts_cano"][base_index:goal_index].copy()
        sampled_r = process_data["right_hand_sampled_verts_cano"][base_index:goal_index].copy()  
        keypoints_concat = np.concatenate((sampled_l, sampled_r), axis=1)
        action = keypoints_concat.reshape(keypoints_concat.shape[0], -1) # [ph, 300]
        return action
    
    def get_dirvec_feature(self, base_index, goal_index, process_data, obj_cano_verts_dense):
        sampled_l = process_data["left_hand_sampled_verts_cano"][base_index:goal_index]
        sampled_r = process_data["right_hand_sampled_verts_cano"][base_index:goal_index]                                         
        # build a KD tree for object in canonical space
        obj_cano_verts_dense = obj_cano_verts_dense[base_index:goal_index]
        trees = [cKDTree(b) for b in obj_cano_verts_dense] # rebuild tree after subsample
        # get the closest point on object for each hand keypoint
        dirvec_concat = []
        for t in range(len(sampled_l)):
            dist_l, point_idx_l = trees[t].query(sampled_l[t],1)
            dist_r, point_idx_r = trees[t].query(sampled_r[t],1)
            # compute dirvec 
            left_distvec = obj_cano_verts_dense[t][point_idx_l] - sampled_l[t]
            right_distvec = obj_cano_verts_dense[t][point_idx_r] - sampled_r[t]
            dirvec_concat.append(np.concatenate((left_distvec, right_distvec), axis=0))
        return np.array(dirvec_concat).reshape(len(dirvec_concat), -1) # ph, 600



    def load_obj_mano_files(self):

        self.category_list = ["box", "espressomachine", "ketchup", 
                    "laptop", "microwave", "mixer", "notebook", "phone", 
                    "scissors", "waffleiron", "capsulemachine"] # for onehot mask
        print("category list contains:")
        print(self.category_list)

        self.data_files = {}
        self.data_files["obj_files"] = []
        self.data_files["mano_files"] = []
        # load original mesh dict
        self.mesh_dict = data_util.load_mesh_dict(self.base_dir, self.category_list)
    
        for c in self.category_list:
            obj_files = sorted(glob.glob(os.path.join(self.base_dir,"raw_seqs/"+"*/*%s*.object.npy"%c)))
            print("number of files for using objects:%d"%len([s for s in obj_files if "use" in s]))
            print("number of files for grabing objects:%d"%len([s for s in obj_files if "grab" in s]))
            mano_files = sorted(glob.glob(os.path.join(self.base_dir,"raw_seqs/"+"*/*%s*.mano.npy"%c)))
            self.data_files["obj_files"].extend(obj_files)
            self.data_files["mano_files"].extend(mano_files)


if __name__ == "__main__":
    preprocess = HandFeaturePreprocess()
    print("Finished Hand Feature Preprocessing")

