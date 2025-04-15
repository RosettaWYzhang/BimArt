
import glob
import os
import sys
import numpy as np
from torch.utils.data import Dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
import utils.mano_utils as mano_utils
import utils.data_util as data_util
from utils.data_util import load_unit_mesh_dict

class MotionDataset(Dataset):
    def __init__(self, cfg, split, return_aux_info=True):
        '''
        Args:
            cfg: configuration file
            split: train or test
            return_aux_info: whether to return auxialary information for visualization
                             not relevant for training split
        '''
        self.cfg = cfg
        self.return_aux_info = return_aux_info
        self.base_dir = cfg["data"]["base_dir"]
        self.pred_horizon = cfg["data"]["pred_horizon"]
        self.base_frame = cfg["data"]["base_frame"]
        self.end_frame = cfg["data"]["end_frame"]   
        self.bps_feature_dim  = cfg["data"]["bps_feature_dim"]
        self.split = split 
        self.get_train_test_split()
        self.num_bps_points = cfg["data"]["num_bps_points"] * 2 # because we are doing part-wise
        self.device = "cuda"
        self.left_hand_index = np.load(cfg["data"]["hand_index_path"]).astype(np.int32)
        self.right_hand_index = self.left_hand_index
        self.mano_layer = mano_utils.create_mano_layer()
        # obj state index for translation
        self.obj_t = 4
        self.obj_raw_dim = 7
        self.load_all_files()
        print("Finish Data Initialization")


    def __len__(self):
        return len(self.indices)

    
    def subsample(self, base_index, goal_index, filename, process_data, obj_process_data):
        '''cut the entire sequence into subsequences according to predict horizon

        return data, which is a list of dictionaries
        this function is called during get_item()

        Args:
            base_index: the starting index of the sequence
            goal_index: the ending index of the sequence
            process_data: a dictionary containing all processed data
      

        keys in self.data: 
            for network training: "action", "obs"
            auxiliary information: "viz"
        '''
        index_list = np.arange(base_index, goal_index, 1)
        # ****************************************** Processing Compulsory Action Features ********************************
        seq_dict = {}
        # ****************************************** Processing Action Feature ********************************************
        seq_dict["action"] = process_data["kp"][index_list]
        dirvec = process_data["dirvec"][index_list]
        seq_dict["action"] = np.concatenate((seq_dict["action"], dirvec), axis=-1)
        # ****************************************** Processing Condition Features as Condition ***************************
        seq_dict["obs"] = {}
        file_cat = process_data["category"]
        bps_feature = obj_process_data["obj_cano_bps"][index_list].copy()
        contact_feature = self.get_contact_features(index_list, obj_process_data)
        seq_dict["obs"]["contact_points"] = contact_feature
        seq_dict["obs"]["object"] = bps_feature.reshape(-1, self.bps_feature_dim * self.cfg["data"]["num_bps_points"] * 2) # bs, 1024*bps_dim
        seq_dict["obs"]["global_states"] = self.get_global_features(index_list, file_cat, obj_process_data)
        # ************************** Store Auxialary Information for Visualization / Processing back to World Space *********************
        if self.return_aux_info:
            seq_dict["viz"] = {}
            seq_dict["viz"]["category"] = file_cat
            seq_dict["viz"]["filename"] = filename
            seq_dict["viz"]["left"] = {}
            seq_dict["viz"]["right"] = {}
            seq_dict["viz"]["start_index"] = base_index
            seq_dict["viz"]["end_index"] = goal_index
            seq_dict["viz"]["left"]["hand_verts"] = process_data["left_hand_verts"][index_list]#
            seq_dict["viz"]["right"]["hand_verts"] = process_data["right_hand_verts"][index_list]#
            seq_dict["viz"]["left"]["hand_keypoints"] = process_data["left_hand_keypoints"][index_list]#
            seq_dict["viz"]["right"]["hand_keypoints"] = process_data["right_hand_keypoints"][index_list]#
            seq_dict["viz"]["left"]["hand_sampled_verts"] = process_data["left_hand_sampled_verts"][index_list]#
            seq_dict["viz"]["right"]["hand_sampled_verts"] = process_data["right_hand_sampled_verts"][index_list]#
            bps_inds = obj_process_data["obj_cano_bps_inds"][index_list] # ph, 1024
            bps_feature = obj_process_data["obj_cano_bps"][index_list]
            seq_dict["viz"]["obj_world_state"] = obj_process_data["obj_world_state"][index_list].copy() 
            seq_dict["viz"]["mano_dict"] = data_util.slice_manodict(process_data["mano_dict"], base_index, goal_index)
            seq_dict["viz"]["bps_viz_index"] = bps_inds
            seq_dict["viz"]["bps_feature"] = bps_feature
            seq_dict["viz"]["gt_contact"] = contact_feature
            seq_dict["viz"]["action"] = process_data["kp"][index_list] 
            seq_dict["viz"]["action"] = np.concatenate((process_data["kp"][index_list], process_data["dirvec"][index_list]), axis=-1) 
        return seq_dict
    
    def get_global_features(self, index_list, file_cat, obj_process_data):
        '''
        get global features of an object, which includes rotation, translation, and scale
        Returns:
            global_states: T X 7
        '''
        global_states = obj_process_data["obj_world_state"][index_list][:, 1:].copy()
        prev_index = index_list[0] - 1
        global_rot = global_states[:, :3]
        global_trans = global_states[:, 3:]
        init_trans = obj_process_data["obj_world_state"][prev_index][4:].copy()
        rel_trans = global_trans - init_trans
        global_states = np.concatenate((global_rot, rel_trans), axis=-1)
        scale = self.mesh_dict[file_cat]["scale"]
        scale = np.array(scale).reshape(1, -1)
        # repeat for all frames
        scale = np.repeat(scale, len(index_list), axis=0)
        global_states = np.concatenate((global_states, scale), axis=-1)
        return global_states

    def get_contact_features(self, index_list, obj_process_data):
        '''
        Get contact distance according to BPS index

        Args:
            index_list: list of indices for subsampling / truncation 
            obj_process_data: processed object data
        Returns:
            contact_feature: contact feature for left and right hand
        '''
        curr_bps_inds = obj_process_data["obj_cano_bps_inds"][index_list]
        left_hand_contact_dense = obj_process_data["contact_dict"]["left"]["dist"][index_list]
        right_hand_contact_dense = obj_process_data["contact_dict"]["right"]["dist"][index_list]
        left_hand_contact = np.zeros((len(index_list), self.num_bps_points))
        right_hand_contact = np.zeros((len(index_list), self.num_bps_points))
        # creating a sparse contact featrue according to BPS indices
        for i in range(len(index_list)):
            left_hand_contact[i] = left_hand_contact_dense[i,curr_bps_inds[i]]
            right_hand_contact[i] = right_hand_contact_dense[i,curr_bps_inds[i]]
        return np.concatenate([left_hand_contact, right_hand_contact], axis=-1)

    def generate_file_indices(self, file_idx, seq_len, filename, split, subject_name):
        '''generate subsequence indices for training and testing'''
        file_indices = []
        if self.end_frame is None:
            # chop the last base_frame so that T pose is not included
            end_frame = seq_len-self.pred_horizon-self.base_frame
        else:
            end_frame = self.end_frame

        if split == "train":
            # sliding window for training by 1 frame
            for subsequence_idx in range(self.base_frame, end_frame):
                file_indices.append({'file_idx': file_idx, 'subsequence_idx': subsequence_idx, 
                                     'filename': filename, 'subject_name': subject_name})
        else:
            # for test, non-overlapping subsequences
            start_frame = self.base_frame
            while start_frame + self.pred_horizon < end_frame:
                file_indices.append({'file_idx': file_idx, 'subsequence_idx': start_frame, 
                                     'filename': filename, 'subject_name': subject_name})
                start_frame += self.pred_horizon

        print("For file idx %d total number of subsequences is %d" %(file_idx, len(file_indices)))
        return file_indices


    def load_all_files(self):
        process_data_all = []
        obj_process_data_all = []

        indices = []
        for file_idx in range(len(self.data_files[self.split]["hand_process_files"])):
            filename = self.data_files[self.split]["hand_process_files"][file_idx].split("/")[-1].split("_processed_hand_features")[0]
            process_data = np.load(self.data_files[self.split]["hand_process_files"][file_idx], allow_pickle=True).item()
            seq_len = process_data["left_hand_sampled_verts"].shape[0]
            subject_name = self.data_files[self.split]["hand_process_files"][file_idx].split("/")[-2]
            file_indices = self.generate_file_indices(file_idx, seq_len, filename, split=self.split, subject_name=subject_name)
            process_data_all.append(process_data)   
            indices.extend(file_indices)
            obj_process_data_all.append(np.load(self.data_files[self.split]["obj_process_files"][file_idx], allow_pickle=True).item())

        self.process_data_all = process_data_all
        self.obj_process_data_all = obj_process_data_all

        assert len(self.process_data_all) == len(self.obj_process_data_all)
        print("************** total number of proprocessed files is %d ****************" %len(self.process_data_all))
        if self.split == "train":
            np.random.shuffle(indices)
        print("!!!total number of subsequences is %d" %len(indices))
        self.indices = indices


    def __getitem__(self, idx):
        index_info = self.indices[idx]
        file_idx = index_info['file_idx']
        start_index = index_info['subsequence_idx']
        subject = index_info['subject_name']
        filename = index_info['filename']
        filename = f"{filename}_{subject}_{start_index}"
        process_data = self.process_data_all[file_idx]
        obj_process_data = self.obj_process_data_all[file_idx]
        end_index = start_index + self.pred_horizon
        nbatch = self.subsample(start_index, end_index, filename, process_data, obj_process_data)
        return nbatch
  


    def get_train_test_split(self):
        '''
        Load file paths belonging to train or test split
        '''
        self.category_list = ["box", "espressomachine", "ketchup", 
                "laptop", "microwave", "mixer", "notebook", "phone", 
                "scissors", "waffleiron", "capsulemachine"]
        print("category list contains:")
        print(self.category_list)
        self.data_files = {}
        self.data_files["train"] = {}
        self.data_files["test"] = {}
        self.data_files["train"]["hand_process_files"] = []
        self.data_files["test"]["hand_process_files"] = []
        self.data_files["train"]["obj_process_files"] = []
        self.data_files["test"]["obj_process_files"] = []
        self.mesh_dict = load_unit_mesh_dict()

        for c in self.category_list:
            print("loading category %s" %c)
            cat_dir = os.path.join(self.base_dir, c)
            obj_process_files = sorted(glob.glob("%s/*/"%cat_dir + "*processed_obj_features.npy"))
            process_files = sorted(glob.glob("%s/*/"%cat_dir + "*processed_hand_features.npy"))               
            test_idx = [0, -1, -2, -3] 
            # wait until data is all processed for better testing
            process_files_test = sorted(list(np.array(process_files)[test_idx]))
            obj_process_files_test = sorted(list(np.array(obj_process_files)[test_idx]))
            process_files_train = sorted(list(set(process_files) - set(process_files_test)))
            obj_process_files_train = sorted(list(set(obj_process_files) - set(obj_process_files_test)))
            self.data_files["train"]["hand_process_files"].extend(process_files_train)
            self.data_files["test"]["hand_process_files"].extend(process_files_test)
            self.data_files["train"]["obj_process_files"].extend(obj_process_files_train)
            self.data_files["test"]["obj_process_files"].extend(obj_process_files_test)

        print("Overall, train file number is %d, test file number is %d" %(len(self.data_files["train"]["obj_process_files"]),
                                                                            len(self.data_files["test"]["obj_process_files"])))

