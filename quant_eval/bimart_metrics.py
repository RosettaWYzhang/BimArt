'''Running BimArt metrics on saved results.

Example command:
python quant_eval/bimart_metrics.py --experiment_dir experiments/motion_model/ --npy_dir eval_output/test_inference

'''

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils import metric_util
import numpy as np
import datetime
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--experiment_dir", type=str, 
                       help="base path of the experiment, computed metrics will also be saved here", 
                       required=True)
argparser.add_argument("--npy_dir", type=str, 
                       help="where npy files to evaluate are saved, should be a subfolder under experiment_dir", 
                       required=True)
argparser.add_argument("--category", type=str, default="all", help="category to evaluate, by default all test sequences are evaluated")
argparser.add_argument("--run_multimodality", action=argparse.BooleanOptionalAction, default=False)
argparser.add_argument("--run_accel_metric", action=argparse.BooleanOptionalAction, default=True)
argparser.add_argument("--run_penetration_1cm", action=argparse.BooleanOptionalAction, default=True)
argparser.add_argument("--run_contact", action=argparse.BooleanOptionalAction, default=True)
argparser.add_argument("--run_articulation", action=argparse.BooleanOptionalAction, default=True)
argparser.add_argument("--key", default="pred", type=str, help="key to use for evaluation, pred or gt")
argparser.add_argument("--prefix", default="", type=str, help="prefix for the saved file")
args = argparser.parse_args()
print(args)


def run_multi(key="pred"):
    print("********************* Multimodality *********************")
    apd_metric_multi = metric_util.APDMetric()
    multi_start_time = datetime.datetime.now()
    visited_batch_name = []
    batch_count = 0
    # filter samples with the same sequence name
    for f in files:
        batch_name = f.split("batch")[0]
        if batch_name not in visited_batch_name:
            visited_batch_name.append(batch_name)
            # grab all files names with the same batch name
            batch_files = [x for x in files if batch_name in x]
            print("File %d: %d batch files found for sequence name : " %(batch_count, len(batch_files)), batch_name)
            batch_count += 1
            # read data from batch files
            pred_verts = []
            for file in batch_files:
                data = np.load(os.path.join(foldername, file), allow_pickle=True).item()
                left_verts_pred = data[key]["left"]["verts"].squeeze()
                right_verts_pred = data[key]["right"]["verts"].squeeze()
                verts_pred = np.concatenate([left_verts_pred, right_verts_pred], axis=1)
                pred_verts.append(verts_pred)
            pred_verts = np.array(pred_verts) # (5, 64, 1556, 3) # (num_sample, num_frames, num_vertices, 3)
            apd_metric_multi.update(pred_verts)
    print("multi modality metric: ", apd_metric_multi.compute())
    multi_end_time = datetime.datetime.now()
    print('Processing time: ', multi_end_time - multi_start_time)
    return apd_metric_multi.compute()

def run_accel(key="pred"):
    # I used jitter interchangably with acceleration
    print("********************* Acceleration *********************")
    jitter_metric = metric_util.JitterMetric()
    vel_start_time = datetime.datetime.now()
    visited_batch_name = []
    for f, file in enumerate(files):
        batch_name = file.split("run")[0]
        if batch_name not in visited_batch_name:
            visited_batch_name.append(batch_name)
            if f % 500 == 0:
                print('Processing file %d out of %d: ' %(f, total_len))
            data = np.load(os.path.join(foldername, file), allow_pickle=True).item()
            # get the ground truth and predicted data
            left_verts_pred = data[key]["left"]["verts"].squeeze()
            right_verts_pred = data[key]["right"]["verts"].squeeze()
            # concatenate left with right
            verts_pred = np.concatenate([left_verts_pred, right_verts_pred], axis=1)
            jitter_metric.update(verts_pred)
            # dimension: (num_frames, num_vertices, 3)
    print("Accel error: ", jitter_metric.compute())
    end_time = datetime.datetime.now()
    print('Processing time: ', end_time - vel_start_time)  
    return jitter_metric.compute()


def run_pen_1cm(key="pred"):
    print("********************* Penetration with 1cm Threshold *********************")
    PenetrationDepthMetric = metric_util.PenetrationDepthMetric(thres=0.01)
    pen_start_time = datetime.datetime.now()
    visited_batch_name = []
    for f, file in enumerate(files):
        batch_name = file.split("run")[0]
        if batch_name not in visited_batch_name:
            visited_batch_name.append(batch_name)
            if f % 500 == 0:
                print('Processing file %d out of %d: ' %(f, total_len))
            # load pickle file
            data = np.load(os.path.join(foldername, file), allow_pickle=True).item()
            # get the ground truth and predicted data
            right_verts_pred = data[key]["right"]["verts"].squeeze()
            obj_verts = data["obj_verts"].squeeze()
            obj_faces = data["obj_faces"]
            # concatenate left with right
            left_verts_pred = data[key]["left"]["verts"].squeeze()
            verts_pred = np.concatenate([left_verts_pred, right_verts_pred], axis=1)
            PenetrationDepthMetric.update(verts_pred, obj_verts, obj_faces)
    print('Penetration error for 1cm: ', PenetrationDepthMetric.compute())
    end_time = datetime.datetime.now()
    print('Processing time: ', end_time - pen_start_time)
    return PenetrationDepthMetric.compute()


def run_contact(key="pred"):
    print("********************* Contact Percentage *********************")
    contact_metric = metric_util.ContactPercentageMetric()
    contact_start_time = datetime.datetime.now()
    visited_batch_name = []
    for f, file in enumerate(files):
        batch_name = file.split("run")[0]
        if batch_name not in visited_batch_name:
            visited_batch_name.append(batch_name)
            if f % 500 == 0:
                print('Processing file %d out of %d: ' %(f, total_len))
            # load pickle file
            data = np.load(os.path.join(foldername, file), allow_pickle=True).item()
            # get the ground truth and predicted data
            right_verts_pred = data[key]["right"]["verts"].squeeze()
            obj_verts = data["obj_verts"].squeeze()
            obj_faces = data["obj_faces"]
            # concatenate left with right
            left_verts_pred = data[key]["left"]["verts"].squeeze()
            verts_pred = np.concatenate([left_verts_pred, right_verts_pred], axis=1)
            contact_metric.update(verts_pred, obj_verts, obj_faces)
    print('Contact percentage: ', contact_metric.compute())
    print("Total number of frames: ", contact_metric.total)
    print("Total number of frames with contact: ", contact_metric.con)

    end_time = datetime.datetime.now()
    print('Processing time: ', end_time - contact_start_time)
    return contact_metric.compute()


def run_arti(key="pred"): 
    print("********************* Articulation *********************")
    arti_metric = metric_util.ArtiContactPercentageMetric()
    arti_start_time = datetime.datetime.now()
    visited_batch_name = []
    for f, file in enumerate(files):
        batch_name = file.split("run")[0]
        if batch_name not in visited_batch_name:
            visited_batch_name.append(batch_name)
            if f % 500 == 0:
                print('Processing file %d out of %d: ' %(f, total_len))
            # load pickle file
            data = np.load(os.path.join(foldername, file), allow_pickle=True).item()
            right_verts_pred = data[key]["right"]["verts"].squeeze()
            obj_verts = data["obj_verts"].squeeze()
            obj_faces = data["obj_faces"]
            # concatenate left with right
            left_verts_pred = data[key]["left"]["verts"].squeeze()
            verts_pred = np.concatenate([left_verts_pred, right_verts_pred], axis=1)
            arti_metric.update(verts_pred, obj_verts, obj_faces, data["obj_parts"].squeeze(), data["obj_gt_states"])
    print('Articulation percentage: ', arti_metric.compute())
    print("Total number of frames with articulation: ", arti_metric.total)
    print("Total number of frames with articulation contact ", arti_metric.con)
    end_time = datetime.datetime.now()
    print('Processing time: ', end_time - arti_start_time)
    return arti_metric.compute()



if __name__ == "__main__":
    foldername = os.path.join(args.experiment_dir, args.npy_dir)
    print("evaluating the following folder: ", foldername)
    # get all filenamae in the folder
    files = os.listdir(foldername)
    # filter files if category is specified
    if args.category != "all":
        files = [x for x in files if args.category in x]
    print("category: ", args.category)
    print("number of files: ", len(files))

    # load all metrics 
    total_len = len(files)
    start_time = datetime.datetime.now()

  
    if args.run_accel_metric:
        jitter = run_accel(args.key)
    else:
        jitter = None
    if args.run_penetration_1cm:
        pen1cm = run_pen_1cm(args.key)
    else:
        pen1cm = None
    if args.run_multimodality:
        mul = run_multi(args.key)
    else:
        mul = None
    if args.run_contact:
        con = run_contact(args.key)
    else:
        con = None
    if args.run_articulation:
        arti = run_arti(args.key)
    else:
        arti = None

    stat = {}
    stat["APD_multi"] = mul
    stat["Penetration_1cm"] = pen1cm
    stat["Accel"] = jitter
    stat["Contact"] = con
    stat["Articulation"] = arti
    stat["Num_Batches"] = len(files)
    stat["Category"] = args.category    

    # save as txt
    prefix = args.prefix
    metric_dir = os.path.join(args.experiment_dir, "metrics", prefix)
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir, exist_ok=True)
        print("creating directory: " + metric_dir)
    with open(os.path.join(metric_dir, "%s_%s.txt" %(args.category, prefix)), "w") as f:
        for key, value in stat.items():
            print("%s: %s" %(key, value))
            f.write("%s: %s\n" %(key, value))

    print("Total processing time: ", datetime.datetime.now() - start_time)

