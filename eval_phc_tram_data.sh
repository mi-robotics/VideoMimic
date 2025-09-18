cd PHC

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/

python phc/run_hydra.py \
    learning=im_mcp \
    exp_name=phc_kp_mcp_iccv \
    epoch=-1 \
    test=True \
    env=env_im_getup_mcp \
    robot=smpl_humanoid \
    robot.freeze_hand=True \
    robot.box_body=False \
    env.z_activation=relu \
    env.motion_file=/home/mcarroll/Documents/cd-2/VideoMimic/src/videomimic/data/video_mimic_demos/phc/tram_data.pkl \
    env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth'] \
    env.num_envs=10 \
    env.obs_v=7  \
    im_eval=True \
    headless=False