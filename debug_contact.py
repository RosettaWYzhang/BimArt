import numpy as np
file1 = "/CT/wzhang4/work/from_scratch/dymart_preprocess/box/s01/box_grab_01_processed_data_part_unit_bps.npy"
file2 = "/CT/wzhang_video/work/bimart_code_open/data/arctic_processed_data/box/s01/box_grab_01_processed_obj_features.npy"
# "obj_cano_bps"
data1 = np.load(file1, allow_pickle=True).item()
data2 = np.load(file2, allow_pickle=True).item()
# "obj_cano_bps" are almost the same

print(np.allclose(data1["obj_cano_bps_inds"], data2["obj_cano_bps_inds"]))
