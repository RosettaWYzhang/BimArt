## This repostory contains code and data instructions for [BimArt: A Unified Approach for the Synthesis of 3D Bimanual Interaction with Articulated Objects](https://vcai.mpi-inf.mpg.de/projects/bimart/).


![Alt text](assets/kitchen_scene.gif)

Please follow the instruction below to set up the environment and run our code. 
If you have any question, please either raise a Github issue or contact the first author (wzhang@mpi-inf.mpg.de) directly. 

# Setup Instruction
## Step 1: Clone the Repository
```
git clone https://github.com/RosettaWYzhang/BimArt.git
cd BimArt
```
## Step 2: download MANO
Download MANO according to instruction [here](https://mano.is.tue.mpg.de/download.php). 
Place unzip it and place the mano_v1_2 folder inside BimArt/assets/
Rename the folder with the following command:
```
mv assets/mano_v1_2/models assets/mano_v1_2/mano
```

## Step 3: download the ARCTIC dataset

Download the ARCTIC dataset following the instruction from [here](https://github.com/zc-alexfan/arctic).   
Rename the folder and place it under BimArt/data/arctic_raw.  
The folder structure should look like:
```
BimArt/data/arctic_raw
                      /meta
                      /raw_seqs         
```

## Step 4: ARCTIC data processing
We will next process the raw data before training BimArt. Please run the commands below:
```
python preprocess/hand_feature_preprocess.py --data_base_dir data/arctic_raw
python preprocess/obj_feature_preprocess.py --data_base_dir data/arctic_raw
```
The processed data is 56 GB.
The processed data is saved under BimArt/data/arctic_processed_data. 

## Step 5: Create New Conda Environment 
```
conda create --name bimart_env python=3.10.14
conda activate bimart_env
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install -r requirements.txt
```

## Step 6 (Optional): Training the Motion Model and the Contact Model
You can directly run inference using our provided model checkpoints, please see [Step 8](#step-8-bimart-inference).  
Alternative, you can also train from scratch.  
Optional: to use Weights&Biases, please update your credentials in config_files/train_contact_config.yaml and config_files/train_motion_config.yaml.   
Please set:
```   
use_wandb: true
wandb_pj_name: bimart_motion_model
wandb_entity: enter_your_credential
```
To train the motion model:
```
python train_motion/train_motion_model.py
```
The training takes around 10 hours on a single A40 GPU.

To train the contact model:
```
python contact_prior/train_contact_model.py
```
The training takes around 12 hours on a single A40 GPU.

## Step 7: BimArt Inference
You can run inference using our provided model checkpoints.   
To generate a sample for a test trajectory, run the following command: 
```
python inference/bimart_inference.py --target_filename espressomachine_use_04_s09_228
```
It takes around 5 minutes per sample.  
To generate more samples per test sequence, update n_sample in config_files/bimart_inference.yaml.  
The visualizations will be saved in experiments/{motion_exp_name}/html/  
The output dictionary will be saved in experiments/{motion_exp_name}/eval_output/  
for target_filename, choose any sequences from our test sequence names in assets/test_filenames.txt  
if target_filename is not specified, it will run all test sequences and save the visualizations and output motions.  

# License
Our source code is under the terms of the [Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
This project is only for research or education purposes, and not freely available for commercial use or redistribution.

# Acknowledgement
This work was supported by the Saarbrücken Research Center for Visual Computing, Interaction and Artificial Intelligence (VIA) and the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – GRK 2853/1 “Neuroexplicit Models of Language, Vision, and Action”. The authors would like to thank [Krzysztof Wolski](https://people.mpi-inf.mpg.de/alumni/d4/2025/kwolski/) for the help on Blender visualizations and [Hui Zhang](https://zdchan.github.io/) for the helpful discussion on setting up the [ArtiGrasp](https://github.com/zdchan/artigrasp) baseline.

Part of the code is borrowed or adapted from [MDM](https://github.com/GuyTevet/motion-diffusion-model) and [ObMan](https://github.com/hassony2/obman_train). We thank the authors for open-sourcing their code. 

# Citation
```
@article{zhang2025bimart,
  title={BimArt: A Unified Approach for the Synthesis of 3D Bimanual Interaction with Articulated Objects},
  author={Zhang, Wanyue and Dabral, Rishabh and Golyanik, Vladislav and Choutas, Vasileios and Alvarado, Eduardo and Beeler, Thabo and Habermann, Marc and Theobalt, Christian},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}