import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch 
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import joblib
from tqdm import tqdm
import argparse
import cv2
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--process_split", type=str, default="train")
    parser.add_argument("--upright_start", action="store_true", default=False)
    parser.add_argument("--yaml_path", type=str, default="")

    args = parser.parse_args()


    # Load the yaml file if provided
    yaml_data = None
    if args.yaml_path:
        with open(args.yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)

    #create a dict from the yaml where each key is the file name
    hml3d = {}
    hml3d_labels_out = {}
    for m in yaml_data['motions']:
        splits = m['file'].replace('-smpl', '').split('/')
        key_name_dump = "0-" + "_".join(splits).replace(".npy", "")
        hml3d[key_name_dump] = m

    
    process_split = args.process_split
    upright_start = args.upright_start
    robot_cfg = {
            "mesh": False,
            "rel_joint_lm": True,
            "upright_start": upright_start,
            "remove_toe": False,
            "real_weight": True,
            "real_weight_porpotion_capsules": True,
            "real_weight_porpotion_boxes": True, 
            "replace_feet": True,
            "masterfoot": False,
            "big_ankle": True,
            "freeze_hand": False, 
            "box_body": False,
            "master_range": 50,
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
            "model": "smpl",
        }

    smpl_local_robot = LocalRobot(robot_cfg,)
    if not osp.isdir(args.path):
        print("Please specify AMASS data path")
        import ipdb; ipdb.set_trace()
        
    all_pkls = glob.glob(f"{args.path}/**/*.npz", recursive=True)
    amass_occlusion = joblib.load("sample_data/amass_copycat_occlusion_v3.pkl")
    amass_full_motion_dict = {}
    amass_splits = {
        'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
        'test': ['Transitions_mocap', 'SSM_synced'],
        'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'KIT',  'EKUT', 'TCD_handMocap', "BMLhandball", "DanceDB", "ACCAD", "BMLmovi", "BioMotionLab_NTroje", "Eyes_Japan_Dataset", "DFaust_67"]   # Adding ACCAD
    }
    process_set = amass_splits[process_split]
    length_acc = []
    for data_path in tqdm(all_pkls):
        bound = 0
        splits = data_path.split("/")[5:]
     
        key_name_dump = "0-" + "_".join(splits).replace(".npz", "")

        if key_name_dump not in hml3d:
            continue
   
        hlm3d_motion = hml3d[key_name_dump]

        if (not splits[0] in process_set):
            continue
        
        if key_name_dump in amass_occlusion:
            issue = amass_occlusion[key_name_dump]["issue"]
            if (issue == "sitting" or issue == "airborne") and "idxes" in amass_occlusion[key_name_dump]:
                bound = amass_occlusion[key_name_dump]["idxes"][0]  # This bounded is calucaled assuming 30 FPS.....
                if bound < 10:
                    print("bound too small", key_name_dump, bound)
                    continue
            else:
                print("issue irrecoverable", key_name_dump, issue)
                continue
            
        entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
        
        if not 'mocap_framerate' in  entry_data:
            continue
        framerate = entry_data['mocap_framerate']

        assert framerate == hlm3d_motion['fps']

        if "0-KIT_442_PizzaDelivery02_poses" == key_name_dump:
            bound = -2
        
        skip = int(framerate/30)

        for s_i in range(len(hlm3d_motion['sub_motions'])):

            root_trans = entry_data['trans'][::skip, :]
            pose_aa = np.concatenate([entry_data['poses'][::skip, :66], np.zeros((root_trans.shape[0], 6))], axis = -1)
        
            start_index = int((30) * hlm3d_motion['sub_motions'][s_i]['timings']['start'])
            end_index = min(len(root_trans),int((30) * hlm3d_motion['sub_motions'][s_i]['timings']['end']))

            root_trans = root_trans[start_index:end_index]
            pose_aa = pose_aa[start_index:end_index]

            betas = entry_data['betas']
            gender = entry_data['gender']
            N = pose_aa.shape[0]
            
            if bound == 0:
                bound = N
                
            root_trans = root_trans[:bound]
            pose_aa = pose_aa[:bound]
            N = pose_aa.shape[0]
            if N < 10:
                continue
        
            smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
            pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
            pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

            beta = np.zeros((16))
            gender_number, beta[:], gender = [0], 0, "neutral"
            # print("using neutral model")
            smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
            smpl_local_robot.write_xml(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
            skeleton_tree = SkeletonTree.from_mjcf(f"phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
            root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                        skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                        torch.from_numpy(pose_quat),
                        root_trans_offset,
                        is_local=True)

            
            if robot_cfg['upright_start']:
                pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)  # should fix pose_quat as well here...

                new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
                pose_quat = new_sk_state.local_rotation.numpy()


            pose_quat_global = new_sk_state.global_rotation.numpy()
            pose_quat = new_sk_state.local_rotation.numpy()
            fps = 30

            new_motion_out = {}
            new_motion_out['pose_quat_global'] = pose_quat_global
            new_motion_out['pose_quat'] = pose_quat
            new_motion_out['trans_orig'] = root_trans
            new_motion_out['root_trans_offset'] = root_trans_offset
            new_motion_out['beta'] = beta
            new_motion_out['gender'] = gender
            new_motion_out['pose_aa'] = pose_aa
            new_motion_out['fps'] = fps

            output_key_name = key_name_dump + f"_sample_{s_i}"
            amass_full_motion_dict[output_key_name] = new_motion_out
            hml3d_labels_out[output_key_name] = hlm3d_motion['sub_motions'][s_i]['labels']
        
    # import ipdb; ipdb.set_trace()
    if upright_start:
        print('Len of amass_full_motion_dict', len(amass_full_motion_dict))
        print('Len of hml3d_labels_out', len(hml3d_labels_out))
        joblib.dump(amass_full_motion_dict, "data/amass/amass_train_upright_hml3d.pkl", compress=True)
        joblib.dump(hml3d_labels_out, "data/amass/hml3d_labels.pkl", compress=True)
    else:
        raise ValueError("Not Upright start is not supported")
        joblib.dump(amass_full_motion_dict, "data/amass/amass_train_take6.pkl", compress=True)