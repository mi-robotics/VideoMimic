

import numpy as np
import os
import yaml
from tqdm import tqdm
import os.path as osp

import joblib
import torch
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import torch.multiprocessing as mp
import copy
import gc
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
from scipy.spatial.transform import Rotation as sRot
import random
from phc.utils.flags import flags
from phc.utils.motion_lib_base import MotionLibBase, DeviceCache, compute_motion_dof_vels, FixHeightMode, MotionlibMode
from smpl_sim.utils.torch_ext import to_torch
import glob

USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    
    class Patch:

        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy




class MotionLibSMPL(MotionLibBase):

    def __init__(self, motion_lib_cfg):
        super().__init__(motion_lib_cfg = motion_lib_cfg)
        
        data_dir = "data/smpl"
        
        if osp.exists(data_dir):
            if motion_lib_cfg.smpl_type == "smpl":
                smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
                smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
                smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")
            elif motion_lib_cfg.smpl_type == "smplh":
                smpl_parser_n = SMPLH_Parser(model_path=data_dir, gender="neutral")
                smpl_parser_m = SMPLH_Parser(model_path=data_dir, gender="male")
                smpl_parser_f = SMPLH_Parser(model_path=data_dir, gender="female")
            elif motion_lib_cfg.smpl_type == "smplx":
                smpl_parser_n = SMPLX_Parser(model_path=data_dir, gender="neutral", use_pca=False, create_transl=False, flat_hand_mean = True, num_betas=20)
                smpl_parser_m = SMPLX_Parser(model_path=data_dir, gender="male", use_pca=False, create_transl=False, flat_hand_mean = True, num_betas=20)
                smpl_parser_f = SMPLX_Parser(model_path=data_dir, gender="female", use_pca=False, create_transl=False, flat_hand_mean = True, num_betas=20)
            self.mesh_parsers = {0: smpl_parser_n, 1: smpl_parser_m, 2: smpl_parser_f}
        else:
            print("SMPL models not found, set mesh_parsers to None")
            self.mesh_parsers = None
        
        return
    
    @staticmethod
    def fix_trans_height(pose_aa, trans, curr_gender_betas, mesh_parsers, fix_height_mode):
     
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0

        
        with torch.no_grad():
            frame_check = 30
            gender = curr_gender_betas[0]
            betas = curr_gender_betas[1:]
            mesh_parser = mesh_parsers[gender.item()]
            height_tolorance = 0.0
            vertices_curr, joints_curr = mesh_parser.get_joints_verts(pose_aa[:frame_check], betas[None,], trans[:frame_check])
            
            offset = joints_curr[:, 0] - trans[:frame_check] # account for SMPL root offset. since the root trans we pass in has been processed, we have to "add it back".
            
            if fix_height_mode == FixHeightMode.ankle_fix:
                assignment_indexes = mesh_parser.lbs_weights.argmax(axis=1)
                pick = (((assignment_indexes != mesh_parser.joint_names.index("L_Toe")).int() + (assignment_indexes != mesh_parser.joint_names.index("R_Toe")).int() 
                    + (assignment_indexes != mesh_parser.joint_names.index("R_Hand")).int() + + (assignment_indexes != mesh_parser.joint_names.index("L_Hand")).int()) == 4).nonzero().squeeze()
                diff_fix = ((vertices_curr[:, pick] - offset[:, None])[:frame_check, ..., -1].min(dim=-1).values - height_tolorance).min()  # Only acount the first 30 frames, which usually is a calibration phase.
            elif fix_height_mode == FixHeightMode.full_fix:
                
                diff_fix = ((vertices_curr - offset[:, None])[:frame_check, ..., -1].min(dim=-1).values - height_tolorance).min()  # Only acount the first 30 frames, which usually is a calibration phase.
            
            
            
            trans[..., -1] -= diff_fix
            return trans, diff_fix

    @staticmethod
    def load_motion_with_skeleton(ids, motion_data_list, skeleton_trees, shape_params, mesh_parsers, config, queue, pid):
        # ZL: loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        max_len = config.max_length

        fix_height = config.fix_height
        np.random.seed(np.random.randint(5000)* pid)
        res = {}
        assert (len(ids) == len(motion_data_list))
        
        if pid == 0 and not config.multi_thread:
            pbar = tqdm(range(len(motion_data_list)))
        else:
            pbar = range(len(motion_data_list))
        
        for f in pbar:
            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]
            curr_gender_beta = shape_params[f]

            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            if isinstance(curr_file['root_trans_offset'], torch.Tensor):
                trans = curr_file['root_trans_offset'].clone()[start:end]
            else:
                trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]
 
            pose_aa = to_torch(curr_file['pose_aa'][start:end])
            pose_quat_global = curr_file['pose_quat_global'][start:end]
            

            B, J, N = pose_quat_global.shape

            ##### ZL: randomize the heading ######
            if (not flags.im_eval) and (not flags.test):
                # if True:
                random_rot = np.zeros(3)
                random_rot[2] = np.pi * (2 * np.random.random() - 1.0)
                random_heading_rot = sRot.from_euler("xyz", random_rot)
                pose_aa[:, :3] = torch.tensor((random_heading_rot * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec())
                pose_quat_global = (random_heading_rot * sRot.from_quat(pose_quat_global.reshape(-1, 4))).as_quat().reshape(B, J, N)
                trans = torch.matmul(trans, torch.from_numpy(random_heading_rot.as_matrix().T))
            ##### ZL: randomize the heading ######

            if not mesh_parsers is None:
                trans, trans_fix = MotionLibSMPL.fix_trans_height(pose_aa, trans, curr_gender_beta, mesh_parsers, fix_height_mode = fix_height)
            else:
                trans_fix = 0

            pose_quat_global = to_torch(pose_quat_global)
            sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_trees[f], pose_quat_global, trans, is_local=False)

            curr_motion = SkeletonMotion.from_skeleton_state(sk_state, curr_file.get("fps", 30))
            curr_dof_vels = compute_motion_dof_vels(curr_motion)
            
            if flags.real_traj:
                quest_sensor_data = to_torch(curr_file['quest_sensor_data'])
                quest_trans = quest_sensor_data[..., :3]
                quest_rot = quest_sensor_data[..., 3:]
                
                quest_trans[..., -1] -= trans_fix # Fix trans
                
                global_angular_vel = SkeletonMotion._compute_angular_velocity(quest_rot, time_delta=1 / curr_file['fps'])
                linear_vel = SkeletonMotion._compute_velocity(quest_trans, time_delta=1 / curr_file['fps'])
                quest_motion = {"global_angular_vel": global_angular_vel, "linear_vel": linear_vel, "quest_trans": quest_trans, "quest_rot": quest_rot}
                curr_motion.quest_motion = quest_motion

            curr_motion.dof_vels = curr_dof_vels
            curr_motion.gender_beta = curr_gender_beta
            res[curr_id] = (curr_file, curr_motion)
            
            

        if not queue is None:
            queue.put(res)
        else:
            return res


    
    

class MotionLibSMPLVidMimic(MotionLibSMPL):

    def __init__(self, motion_lib_cfg, extra_data_path):
        self.extra_data_path = extra_data_path
        self.image_emb_mappings = joblib.load(f'{self.extra_data_path}/vidmimic_image_emb_mappings.pkl')
        super().__init__(motion_lib_cfg = motion_lib_cfg)
        
     
        return


    def load_data(self, motion_file,  min_length=-1, im_eval = False):
        # TODO: remove the exlcude failed keys
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))
        
        data_list = self._motion_data_load

        if self.mode == MotionlibMode.file:
            if min_length != -1:
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['pose_quat_global']) >= min_length}
            elif im_eval:
                data_list = {
                        k: v
                        for k, v in sorted(
                            (
                                (k, v)
                                for k, v in self._motion_data_load.items()
                                if isinstance(v, dict)
                                and 'pose_quat_global' in v
                                and v['pose_quat_global'] is not None
                            ),
                            key=lambda kv: len(kv[1]['pose_quat_global']),
                            reverse=True,
                        )
                    }
                # data_list = self._motion_data
            else:
                data_list = self._motion_data_load

            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)
        
        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(self._motion_data_load[0]) 
    
    
    def get_motion_state(self, motion_ids=None, motion_times=None, offset=None):
        motion_state = super().get_motion_state(motion_ids, motion_times, offset)

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)


        # fro each motion get the t-emb and t+8 emb
        
        fetch_keys = self.curr_motion_keys[motion_ids]
   
        all_embs = []
        if len(motion_ids) == 1:
            fetch_keys = [fetch_keys]

        for i, key in enumerate(fetch_keys):

            start_frame_index = (frame_idx0-1)[i]
            end_frame_index = torch.min(frame_idx0 + 7, num_frames - 2)[i]
                        
            image_emb = self.image_emb_mappings[key]

            t_emb = torch.from_numpy(image_emb[start_frame_index]).to(self._device).unsqueeze(0)
            t_plus_8_emb = torch.from_numpy(image_emb[end_frame_index]).to(self._device).unsqueeze(0)

            embs = torch.cat((t_emb, t_plus_8_emb), dim=0)


            all_embs.append(embs.unsqueeze(0))

        motion_state['image_emb'] = torch.cat(all_embs, dim=0)
   
        return motion_state