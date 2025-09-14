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
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState

def convert_tram_to_rl(seq_folder, output_path, upright_start=False):
    """
    Converts TRAM motion capture output to the format used by the RL tracking code.

    Args:
        seq_folder (str): Path to the TRAM sequence folder (containing 'hps', 'images', 'camera.npy').
        output_path (str): Path to save the output .pkl file.
        upright_start (bool): If True, normalizes the motion to start in an upright orientation.
    """
    hps_folder = os.path.join(seq_folder, 'hps')
    hps_files = sorted(glob.glob(os.path.join(hps_folder, '*.npy')))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. Load Camera Data ---
    # This data is crucial for transforming from camera space to world space.
    pred_cam = np.load(os.path.join(seq_folder, 'camera.npy'), allow_pickle=True).item()
    world_cam_R = torch.tensor(pred_cam['world_cam_R'], dtype=torch.float32).to(device)
    world_cam_T = torch.tensor(pred_cam['world_cam_T'], dtype=torch.float32).to(device)
    
    # --- 2. Load Skeleton Tree for Kinematics ---
    # This must point to the SMPL model definition used by your RL code.
    # The path is taken from your provided amass_converter.py script.
    smpl_mjcf_path = "phc/data/assets/mjcf/smpl_humanoid.xml" 
    if not os.path.exists(smpl_mjcf_path):
        raise FileNotFoundError(f"Could not find the SMPL MJCF model at: {smpl_mjcf_path}. Please update the path.")
    skeleton_tree = SkeletonTree.from_mjcf(smpl_mjcf_path)

    tram_full_motion_dict = {}
    seq_name = os.path.basename(seq_folder)

    # --- 3. Process Each Tracked Person ---
    for i, hps_file in enumerate(tqdm(hps_files, desc=f"Processing {seq_name}")):
        pred_smpl = np.load(hps_file, allow_pickle=True).item()
        
        # Load SMPL parameters from the TRAM output
        pred_rotmat = pred_smpl['pred_rotmat'].to(device)           # (num_frames, 24, 3, 3)
        pred_shape = pred_smpl['pred_shape'].to(device)             # (num_frames, 10)
        pred_trans_cam = pred_smpl['pred_trans'].to(device)       # (num_frames, 3) -> In camera space
        frame_indices = pred_smpl['frame']
        
        N = len(frame_indices)
        if N < 10:  # Skip very short tracks
            continue

        # Get the corresponding camera poses for this track's frames
        cam_r_track = world_cam_R[frame_indices] # (N, 3, 3)
        cam_t_track = world_cam_T[frame_indices] # (N, 3)

        # --- 4. Coordinate System Transformation (Camera -> World) ---
        # Transform root translation from camera space to world space
        root_trans_world = torch.einsum('bij,bj->bi', cam_r_track, pred_trans_cam) + cam_t_track
        
        # Transform global orientation from camera space to world space
        global_orient_cam = pred_rotmat[:, 0] # (N, 3, 3)
        global_orient_world = torch.einsum('bij,bjk->bik', cam_r_track, global_orient_cam)
        
        # Combine into a full-body pose in the world frame
        pose_rotmat_world = pred_rotmat.clone()
        pose_rotmat_world[:, 0] = global_orient_world
        
        # --- 5. Convert to Target RL Format ---
        # The first rotation is global orientation, the rest are local rotations
        # Convert rotation matrices to quaternions (N, 24, 4)
        pose_quat = sRot.from_matrix(pose_rotmat_world.cpu().numpy().reshape(-1, 3, 3)).as_quat().reshape(N, 24, 4)

        # Use the average shape for the character
        betas = pred_shape.mean(dim=0).cpu().numpy()
        beta_padded = np.zeros(16)
        beta_padded[:10] = betas
        
        # --- 6. Use poselib to get global and local rotations ---
        # We provide poselib with the world-space root and local joint rotations.
        # It builds the full kinematic state.
        root_trans_offset = root_trans_world.cpu() + skeleton_tree.local_translation[0]
        
        sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True  # True because pose_quat[1:] are local rotations
        )
        
        # Optional: Normalize the starting pose to be upright
        if upright_start:
            # This logic is copied from your amass_converter.py to ensure consistency
            upright_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
            pose_quat_global_upright = (sRot.from_quat(sk_state.global_rotation.reshape(-1, 4).numpy()) * upright_rot).as_quat().reshape(N, -1, 4)
            
            sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree, 
                torch.from_numpy(pose_quat_global_upright), 
                root_trans_offset, 
                is_local=False # Rotations are now global, so is_local=False
            )

        # Extract final data from the skeleton state
        pose_quat_global = sk_state.global_rotation.numpy()
        pose_quat_local = sk_state.local_rotation.numpy()
        
        # Convert local quaternions back to axis-angle for the 'pose_aa' field
        pose_aa = sRot.from_quat(pose_quat_local.reshape(-1, 4)).as_rotvec().reshape(N, -1)
        
        # --- 7. Assemble the final dictionary ---
        key_name_dump = f"{seq_name}_track_{i:02d}"
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
        
    # --- 8. Save the Output ---
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