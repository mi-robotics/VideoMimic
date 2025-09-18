import glob
import os
import sys
import argparse
import joblib
from tqdm import tqdm

import torch
import numpy as np
from scipy.spatial.transform import Rotation as sRot

# Add your project's root directory to the Python path if necessary
# sys.path.append(os.getcwd()) 

# Assuming these libraries are in your environment

PHC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../PHC"))
if PHC_PATH not in sys.path:
    sys.path.insert(0, PHC_PATH)

from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES, SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot


def convert_tram_to_rl(seq_folder, output_path, upright_start=False):
    """
    Converts TRAM motion capture output to the format used by the RL tracking code.

    Args:
        seq_folder (str): Path to the TRAM sequence folder containing subdirectories, 
                         each representing a motion with 'hps', 'images' subdirs and 'camera.npy'.
        output_path (str): Path to save the output .pkl file.
        upright_start (bool): If True, normalizes the motion to start in an upright orientation.
    """
    upright_start = True
    # Validate input directory structure
    if not os.path.isdir(seq_folder):
        raise ValueError(f"Sequence folder does not exist: {seq_folder}")
    
    # Find all motion subdirectories within seq_folder
    motion_dirs = []
    for item in os.listdir(seq_folder):
        motion_path = os.path.join(seq_folder, item)
        if os.path.isdir(motion_path):
            # Check if this subdirectory has the required structure
            hps_folder = os.path.join(motion_path, 'hps')
            images_folder = os.path.join(motion_path, 'images')
            camera_file = os.path.join(motion_path, 'camera.npy')
            
            if (os.path.isdir(hps_folder) and 
                os.path.isdir(images_folder) and 
                os.path.isfile(camera_file)):
                motion_dirs.append(motion_path)
    
    if not motion_dirs:
        raise ValueError(f"No valid motion directories found in: {seq_folder}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. Load Skeleton Tree for Kinematics ---
    # This must point to the SMPL model definition used by your RL code.
    # The path is taken from your provided amass_converter.py script.
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

    smpl_local_robot = LocalRobot(robot_cfg, data_dir="/home/mcarroll/Documents/cd-2/VideoMimic/PHC/data/smpl")
    beta = np.zeros((16))
    gender_number, beta[:], gender = [0], 0, "neutral"
    # print("using neutral model")

    smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
    smpl_local_robot.write_xml(f"{PHC_PATH}/phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
    skeleton_tree = SkeletonTree.from_mjcf(f"{PHC_PATH}/phc/data/assets/mjcf/{robot_cfg['model']}_humanoid.xml")
        

    tram_full_motion_dict = {}
    seq_name = os.path.basename(seq_folder)

    # --- 2. Process Each Motion Directory ---
    for motion_dir in motion_dirs:
        motion_name = os.path.basename(motion_dir)
        print(f"Processing motion: {motion_name}")
        
        # Load Camera Data for this motion
        pred_cam = np.load(os.path.join(motion_dir, 'camera.npy'), allow_pickle=True).item()
        world_cam_R = torch.tensor(pred_cam['world_cam_R'], dtype=torch.float32).to(device)
        world_cam_T = torch.tensor(pred_cam['world_cam_T'], dtype=torch.float32).to(device)
        
        # Get HPS files for this motion
        hps_folder = os.path.join(motion_dir, 'hps')
        hps_files = sorted(glob.glob(os.path.join(hps_folder, '*.npy')))
        
        if not hps_files:
            print(f"Warning: No HPS files found in {hps_folder}, skipping...")
            continue
        
        # Process Only the First Tracked Person ---
        # Only process the first HPS file as requested
        hps_files = hps_files[:1]  # Take only the first file
        for i, hps_file in enumerate(tqdm(hps_files, desc=f"Processing {motion_name}")):
            pred_smpl = np.load(hps_file, allow_pickle=True).item()
        
            # Load SMPL parameters from the TRAM output
       
            pred_rotmat = pred_smpl['pred_rotmat'].to(device).to(torch.float32) # (num_frames, 24, 3, 3)
            pred_shape = pred_smpl['pred_shape'].to(device).to(torch.float32) # (num_frames, 10)
            pred_trans_cam = pred_smpl['pred_trans'].to(device).to(torch.float32) # (num_frames, 1, 3) or (num_frames, 3)
            # Ensure translation shape is (N, 3)
            if pred_trans_cam.dim() == 3 and pred_trans_cam.size(1) == 1:
                pred_trans_cam = pred_trans_cam[:, 0, :]

            # Ensure frame indices are a torch LongTensor on the correct device
            frame_indices_np = pred_smpl['frame']
            frame_indices_t = torch.as_tensor(frame_indices_np, dtype=torch.long, device=device)

            N = frame_indices_t.numel()
            if N < 10:  # Skip very short tracks
                continue
            
            smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
            pred_rotmat = pred_rotmat[:, smpl_2_mujoco] 

            if True:
                # Get the corresponding camera poses for this track's frames
                cam_r_track = world_cam_R[frame_indices_t] # (N, 3, 3)
                cam_t_track = world_cam_T[frame_indices_t] # (N, 3)


                # root_trans_world = torch.einsum('bij,bj->bi', cam_r_track, pred_trans_cam) + cam_t_track
                
                # Transform global orientation from camera space to world space
                global_orient_cam = pred_rotmat[:, 0]  # (N, 3, 3)
                # Multiply camera-to-world rotation with global orientation in camera space to get world orientation
                global_orient_world =  cam_r_track @ global_orient_cam
                
                # Combine into a full-body pose in the world frame
                pose_rotmat_world = pred_rotmat.clone()
                pose_rotmat_world[:, 0] = global_orient_world
            else:
                pose_rotmat_world = pred_rotmat
            root_trans_world = pred_trans_cam
            
            # --- 4. Convert to Target RL Format ---
            # The first rotation is global orientation, the rest are local rotations
            # Convert rotation matrices to quaternions (N, 24, 4)
   
            pose_quat = sRot.from_matrix(pose_rotmat_world.detach().cpu().numpy().reshape(-1, 3, 3)).as_quat().reshape(N, 24, 4)

            # Use the average shape for the character
            betas = pred_shape.mean(dim=0).cpu().numpy()
            beta_padded = np.zeros(16)
            beta_padded[:10] = betas
            
            # --- 5. Use poselib to get global and local rotations ---
            # We provide poselib with the world-space root and local joint rotations.
            # It builds the full kinematic state.
            root_trans_offset = (
                root_trans_world.detach().cpu().to(torch.float32)
                + torch.as_tensor(skeleton_tree.local_translation[0], dtype=torch.float32)
            )
            
            sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,
                torch.from_numpy(pose_quat).to(torch.float32),
                root_trans_offset,
                is_local=True  # True because pose_quat[1:] are local rotations
            )
            # Visualize the skeleton state as a skeleton motion (quick preview)
            # This requires matplotlib and poselib's visualization utilities.
            try:
                import matplotlib.pyplot as plt
                from poselib.poselib.visualization.common import plot_skeleton_motion_interactive, plot_skeleton_states

                # If you want to visualize the motion, you can pass a list of SkeletonState objects (one per frame)
                # or just the SkeletonState object if it represents a sequence.
                # Here, sk_state is a SkeletonState with shape (N, ...), so we can pass it directly.
                sk_motion = SkeletonMotion.from_skeleton_state(sk_state, fps=30)
                # plot_skeleton_states(sk_state, skip_n=1, task_name=f"TRAM Track {i:02d} ({seq_name}/{motion_name})")
                plot_skeleton_motion_interactive(sk_motion)
            except Exception as e:
                print(f"[Visualization skipped] {e}")
            
            # Optional: Normalize the starting pose to be upright
            if upright_start:
                # This logic is copied from your amass_converter.py to ensure consistency
                upright_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                pose_quat_global_upright = (
                    sRot.from_quat(
                        sk_state.global_rotation.detach().cpu().reshape(-1, 4).numpy()
                    )
                    * upright_rot
                ).as_quat().reshape(N, -1, 4)
                
                sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree, 
                    torch.from_numpy(pose_quat_global_upright).to(torch.float32), 
                    root_trans_offset, 
                    is_local=False # Rotations are now global, so is_local=False
                )

            # Extract final data from the skeleton state
            pose_quat_global = sk_state.global_rotation.numpy()
            pose_quat_local = sk_state.local_rotation.numpy()
            
            # Convert local quaternions back to axis-angle for the 'pose_aa' field
            pose_aa = sRot.from_quat(pose_quat_local.reshape(-1, 4)).as_rotvec().reshape(N, -1)
            
            # --- 6. Assemble the final dictionary ---
            key_name_dump = f"{seq_name}_{motion_name}_track_{i:02d}"
            new_motion_out = {
                'pose_quat_global': pose_quat_global.astype(np.float32),
                'pose_quat': pose_quat_local.astype(np.float32),
                'trans_orig': root_trans_world.cpu().numpy().astype(np.float32),
                'root_trans_offset': root_trans_offset.numpy().astype(np.float32),
                'beta': beta_padded.astype(np.float32),
                'gender': "neutral",
                'pose_aa': pose_aa.astype(np.float32),
                'fps': 30  # TRAM typically processes video at 30 FPS
            }
            tram_full_motion_dict[key_name_dump] = new_motion_out
    
    # --- 7. Save the Output ---
    joblib.dump(tram_full_motion_dict, output_path, compress=True)
    print(f"\n✅ Conversion complete! Saved {len(tram_full_motion_dict)} tracks to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TRAM output to RL-ready format.")
    parser.add_argument("--path", type=str, required=True, help="Path to the TRAM sequence folder.")
    parser.add_argument("--output", type=str, required=True, help="Path for the output .pkl file.")
    parser.add_argument("--upright_start", action="store_true", default=False, help="Normalize motion to start upright.")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.path):
        print(f"Error: The specified path '{args.path}' is not a valid directory.")
    else:
        convert_tram_to_rl(args.path, args.output, args.upright_start)