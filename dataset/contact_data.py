
from torch.utils.data import Dataset
import numpy as np
import glob
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils import data_util


class ObjectContactData(Dataset):

    def __init__(self, split="train",
                 base_dir="data/arctic_processed_data",
                 end_frame=None,
                 pred_horizon=64,
                 base_frame=100,
                 return_aux_info=False,
                 ):
        self.base_dir = base_dir
        self.base_frame = base_frame
        self.return_aux_info = return_aux_info
        self.split = split
        self.obj_t = 4  
        self.end_frame = end_frame
        self.pred_horizon = pred_horizon  
        self.mesh_dict = data_util.load_unit_mesh_dict()
        self.__get_file_paths__()
        self.load_data()

    def __getitem__(self, idx):
        start_index = self.indices[idx]['subsequence_idx']
        subject = self.indices[idx]['subject_name']
        filename = self.indices[idx]['filename']
        filename = f"{filename}_{subject}_{start_index}"
        return self.subsample(self.indices[idx]["file_idx"], self.indices[idx]["subsequence_idx"], filename)
    

    def __len__(self):
        return len(self.indices)
    
    def get_frame_contact_feature(self, index, process_data):
        contact_data = process_data["contact_dict"]
        curr_bps_inds = process_data["obj_cano_bps_inds"][index] 
        left_hand_contact = contact_data["left"]["dist"][index[:, np.newaxis], curr_bps_inds]
        right_hand_contact = contact_data["right"]["dist"][index[:, np.newaxis], curr_bps_inds]
        return np.concatenate([left_hand_contact, right_hand_contact], axis=-1)


    def get_frame_obj_feature(self, index, process_data):
        '''
        Args:
            index: frames to be truncated
        return: 
            obj_feat: [seq_len, num_bps*bps_feat]
        '''
        obj_feat = process_data["obj_cano_bps"][index, :].copy()
        obj_feat = obj_feat.reshape(obj_feat.shape[0], -1)
        return obj_feat 
    
    def get_global_states(self, index, process_data, filecat):
        '''
        concatenate rotation, relative translation to the first frame and scale
          to get [seq_len, 6] global states vector
        '''
        curr_global_states = process_data["obj_world_state"][index][:,1:].copy()
        curr_rot = curr_global_states[:, :3].copy()
        rel_trans = curr_global_states[:, 3:] - curr_global_states[0][3:]
        curr_global_states = np.concatenate((curr_rot, rel_trans), axis=-1)
        scale = self.mesh_dict[filecat]["scale"]
        scale = np.array([scale])
        # repeat along the seq_len dimension
        scale = np.repeat(scale, len(index), axis=0)
        curr_global_states = np.concatenate([curr_global_states, scale[:, np.newaxis]], axis=-1)
        return curr_global_states

    
    def subsample(self, file_start_idx, start_idx, filename):
        '''return a dictionary of features for training, 
           and some auxiliary information for visualization
        '''
        input_dict = {}
        input_dict["obs"] = {}
        process_data = self.process_data_all[file_start_idx]
        end_idx = start_idx + self.pred_horizon
        index_list = np.arange(start_idx, end_idx, 1)  
        input_dict["action"] = self.get_frame_contact_feature(index_list, process_data)
        input_dict["obs"]["obj_feat"] = self.get_frame_obj_feature(index_list, process_data)
        filecat = process_data["category"]
        input_dict["obs"]["curr_global_states"] = self.get_global_states(index_list, process_data, filecat)
        # auxiliary features are mostly for visualization
        input_dict["aux"] = {}
        input_dict["aux"]["category"] = process_data["category"]
        input_dict["aux"]["filename"] = filename
        if self.split == "test" and self.return_aux_info:
            input_dict["aux"]["bps_index"] = process_data["obj_cano_bps_inds"][index_list, :].copy().reshape(-1, 1024)
            input_dict["aux"]["obj_gt_state"] = process_data["obj_world_state"][index_list, :].copy().reshape(-1, 7)
        return input_dict
        

    def load_data(self):
        '''load all processed files
        '''
        indices = []
        self.process_data_all = []
        print("loading data for %s split" %self.split)
        for f in range(len(self.data_files[self.split]["process_files"])):
            # load all files
            print("............load file %d...................." %f, flush=True) 
            process_data = np.load(self.data_files[self.split]["process_files"][f], allow_pickle=True).item()
            filename = self.data_files[self.split]["process_files"][f].split("/")[-1].split("_processed_data")[0]
            subject_name = self.data_files[self.split]["process_files"][f].split("/")[-2]
            seq_len = process_data["obj_world_state"].shape[0]
            if self.end_frame is None:
                # chop the last base_frame so that T pose is not included
                end_frame = seq_len-self.pred_horizon-self.base_frame
            else:
                end_frame = self.end_frame
            if self.split == "train":
                for subsequence_idx in range(self.base_frame, end_frame):
                    indices.append({'file_idx': f, 'subsequence_idx': subsequence_idx,
                                    'subject_name': subject_name, 'filename': filename})
            else:
                start_frame = self.base_frame
                while start_frame + self.pred_horizon < end_frame:
                    indices.append({'file_idx': f, 'subsequence_idx': start_frame,
                                    'subject_name': subject_name, 'filename': filename})
                    start_frame += self.pred_horizon
            self.process_data_all.append(process_data)
        np.random.shuffle(indices) # in-place shuffle
        self.indices = indices       
        print("total number of subsequences is " + str(len(indices)))
            

    def __get_file_paths__(self):
        self.category_list = ["box", "espressomachine", "ketchup", 
                "laptop", "microwave", "mixer", "notebook", "phone", 
                "scissors", "waffleiron", "capsulemachine"]
        print("category list contains:")
        print(self.category_list)

        self.data_files = {}
        self.data_files["train"] = {}
        self.data_files["test"] = {}
        self.data_files["train"]["process_files"] = []
        self.data_files["test"]["process_files"] = []
        for c in self.category_list:
            print("loading category %s" %c)
            cat_dir = os.path.join(self.base_dir, c)
            process_files = sorted(glob.glob("%s/*/"%cat_dir + "*processed_obj_features.npy"))
            test_idx = [0, -1, -2, -3] # hard coded
            process_files_test = sorted(list(np.array(process_files)[test_idx]))
            process_files_train = sorted(list(set(process_files) - set(process_files_test)))
            self.data_files["train"]["process_files"].extend(process_files_train)
            self.data_files["test"]["process_files"].extend(process_files_test)

    