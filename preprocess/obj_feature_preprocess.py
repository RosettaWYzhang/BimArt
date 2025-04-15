'''
This script preprocesses object features for the BimArt dataset from the raw arctic data sequences.
'''

import os
import numpy as np
import glob
import sys
from scipy.spatial import cKDTree
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from utils import data_util, mano_utils
import polyscope as ps

parser = argparse.ArgumentParser()
parser.add_argument('--data_base_dir', default="data/arctic_raw",
                    type=str,
                    help='path of the downloaded artic data')
parser.add_argument('--data_process_dir',
                    default="data/arctic_processed_data",
                    type=str, help='path of the processed artic data')
args = parser.parse_args()


class ObjFeaturePreprocess():
    def __init__(self):
        '''
        This class extracts part-based bps features and contact for objects
        '''
        self.file_suffix = "processed_obj_features"
        self.base_dir = args.data_base_dir
        self.bps_points = np.load("assets/bps_normalized_part_based.npy")
        self.mano_layer = mano_utils.create_mano_layer()
        self.load_raw_files()
        self.obj_trans_idx = 4
        self.compute_obj_features()

    def load_raw_files(self):
        '''
        Load object and mano files from raw arctic sequences'''
        self.category_list = ["box", "espressomachine", "ketchup",
                              "laptop", "microwave", "mixer",
                              "notebook", "phone", "scissors",
                              "waffleiron", "capsulemachine"]
        print("category list contains:")
        print(self.category_list)

        self.data_files = {}
        self.data_files["obj_files"] = []
        self.data_files["mano_files"] = []
        # load mesh in original scale
        self.mesh_dict = data_util.load_unit_mesh_dict()
        self.original_mesh_dict = data_util.load_mesh_dict()
        for c in self.category_list:
            obj_files = sorted(glob.glob(
                os.path.join(self.base_dir,"raw_seqs/"+"*/*%s*.object.npy" % c)))
            mano_files = sorted(glob.glob(
                os.path.join(self.base_dir,"raw_seqs/"+"*/*%s*.mano.npy" % c)))
            print("number of files for using objects:%d" % len(
                [s for s in obj_files if "use" in s]))
            print("number of files for grabing objects:%d" % len(
                [s for s in obj_files if "grab" in s]))
            self.data_files["obj_files"].extend(obj_files)
            self.data_files["mano_files"].extend(mano_files)
            assert len(obj_files) == len(mano_files)

    def compute_obj_features(self):
        '''
        Process raw object data into bps features, contact dictionary, and save processed data.
        '''
        print("loading %d files" % len(self.data_files["obj_files"]))
        for f in range(len(self.data_files["obj_files"])):
            file_cat = self.data_files["obj_files"][f].split("/")[-1].split("_")[0]
            seq_name = self.data_files["obj_files"][f].split("/")[-2]
            filename = self.data_files["obj_files"][f].split("/")[-1].split(".")[0] + "_%s.npy" %self.file_suffix
            print("process category %s" % file_cat)
            print("............load file %d...................." % f,
                  flush=True)
            obj_state_div = np.load(self.data_files["obj_files"][f],
                                    allow_pickle=True)
            mano_dict = np.load(self.data_files["mano_files"][f],
                                allow_pickle=True).item()
            obj_state_div[:, self.obj_trans_idx:] /= 1000
            # divide translation, so that it is consistency with hand unit
            obj_state_artionly = obj_state_div.copy()
            obj_state_artionly[:, 1:] = np.zeros_like(obj_state_div[:, 1:])
            obj_cano_verts_dense = data_util.object_verts_to_world_batch(
                self.mesh_dict[file_cat]["verts"],
                self.mesh_dict[file_cat]["parts"],
                state=obj_state_artionly
                )

            obj_part = self.mesh_dict[file_cat]["parts"]
            top_inds = np.where(obj_part == 0)[0]
            bottom_inds = np.where(obj_part == 1)[0]
            trees = [cKDTree(b) for b in obj_cano_verts_dense[:,obj_part == 0]]
            bottom_tree = cKDTree(obj_cano_verts_dense[0,obj_part == 1])
            bps_feature, bps_inds = data_util.get_bps_feature_batch(trees, self.bps_points)
            bottom_bps_feature, bottom_bps_inds = data_util.get_bps_feature_from_tree(
                bottom_tree, self.bps_points)
            bottom_bps_feature = np.repeat(bottom_bps_feature[None, :], len(bps_feature), axis=0)
            bottom_bps_inds = np.repeat(bottom_bps_inds[None, :], len(bps_inds), axis=0)
            bps_feature = np.concatenate((bps_feature, bottom_bps_feature), axis=1)
            top_ori_inds = top_inds[bps_inds]
            bottom_ori_inds = bottom_inds[bottom_bps_inds]
            bps_inds = np.concatenate((top_ori_inds, bottom_ori_inds), axis=1)

            processed_data = {}
            processed_data["category"] = file_cat
            processed_data["obj_cano_verts_dense"] = obj_cano_verts_dense
            processed_data["obj_cano_bps"] = bps_feature
            processed_data["obj_cano_bps_inds"] = bps_inds
            processed_data["obj_world_state"] = obj_state_div
            processed_data["obj_cano_arti_state"] = obj_state_artionly
            # for contact computation, we need object to be at the original scale
            obj_cano_verts_original_scale = data_util.object_verts_to_world_batch(
                self.original_mesh_dict[file_cat]["verts"],
                self.original_mesh_dict[file_cat]["parts"],
                state=obj_state_artionly
                )

            processed_data["contact_dict"] = self.compute_contact_dist(
                obj_state_world=obj_state_div, 
                mano_dict=mano_dict, 
                obj_verts_cano=obj_cano_verts_original_scale)
            
            save_dir = os.path.join(args.data_process_dir, file_cat, seq_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_file_path = os.path.join(save_dir, filename)   
     
            print("saving file %s" % save_file_path)
            np.save(save_file_path, processed_data)
            

    def compute_contact_dist(self, obj_state_world, mano_dict, obj_verts_cano):
        '''
        This function computes the distance between object vertices and hand vertices in canonical space

        Args:
            obj_state_world: np array, object state in world space with shape (T, 7), use for canonicalization
            mano_dict: mano dict with contains numpy arrays of mano parameters
            obj_verts_cano: np array, object vertices in canonical space with shape (T, N, 3), original scale
        Returns:
            dist_map_dict: dict, contains distance and nearest index for left and right hand separately 
        '''
        dist_list_left = []
        nearest_index_left = []
        dist_list_right = []
        nearest_index_right = []

        mano_dict_tensor = data_util.NpDictToTensor(mano_dict)

        # get hand vertices from mano
        verts_left, _ = mano_utils.get_hand_verts_joints(
            self.mano_layer, "left", mano_dict_tensor)
        verst_right, _ = mano_utils.get_hand_verts_joints(
            self.mano_layer, "right", mano_dict_tensor)

        # transform hand keypoints to canonical space
        rot = obj_state_world[:, 1:self.obj_trans_idx]
        trans = obj_state_world[:, self.obj_trans_idx:7]
        # trasnform hand verts to canonical space
        verts_left = data_util.canonicalize_pointcloud(verts_left, trans, rot)
        verts_right = data_util.canonicalize_pointcloud(verst_right, trans, rot)

        for t in range(obj_state_world.shape[0]):
            # build kd tree for hand, instead of objects
            kdtree_left=cKDTree(verts_left[t])
            kdtree_right=cKDTree(verts_right[t])
            dist_left, idx_left = kdtree_left.query(obj_verts_cano[t], 1)
            dist_right, idx_right = kdtree_right.query(obj_verts_cano[t], 1)
            dist_list_left.append(dist_left)
            nearest_index_left.append(idx_left)
            dist_list_right.append(dist_right)
            nearest_index_right.append(idx_right)

        dist_left_dict = {"dist": np.array(dist_list_left), "index": np.array(nearest_index_left)}
        dist_right_dict = {"dist": np.array(dist_list_right), "index": np.array(nearest_index_right)}
        dist_map_dict = {"left": dist_left_dict, "right": dist_right_dict}
        return dist_map_dict




if __name__ == "__main__":
    ObjFeaturePreprocess()
    print("Finished Object Feature Preprocessing")

