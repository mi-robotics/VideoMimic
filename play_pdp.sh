
cd PHC

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/

CUDA_LAUNCH_BLOCKING=1


#T2M
# python scripts/phc_act/play_pdp.py \
#     --dataset_path=data/amass/amass_train_upright.pkl \
#     --exp_name=phc_kp_mcp_iccv \
#     --num_runs=10 \
#     --action_noise_std=0.1 \
#     --pdp_policy_path=/home/mcarroll/Documents/cd-2/VideoMimic/PDP/outputs/2025.09.23/23.11.51_pdp_phc_v2_hml3d_ft/checkpoints/checkpoint_epoch_100.ckpt


    # --pdp_policy_path=/home/mcarroll/Documents/cd-2/VideoMimic/PDP/outputs/2025.09.23/15.55.51_pdp_phc_v2_hml3d_ft/checkpoints/checkpoint_epoch_100.ckpt


#Reference
# python scripts/phc_act/play_pdp.py \
#     --dataset_path=data/amass/amass_train_upright.pkl \
#     --exp_name=phc_kp_mcp_iccv \
#     --num_runs=10 \
#     --action_noise_std=0.1 \
#     --pdp_policy_path=/home/mcarroll/Documents/cd-2/VideoMimic/PDP/outputs/2025.09.25/10.43.53_pdp_phc_v2_ref/checkpoints/checkpoint_epoch_17.ckpt


# VidMimic
python scripts/phc_act/play_pdp.py \
    --dataset_path=data/amass/amass_train_upright_vidmimic.pkl \
    --exp_name=phc_kp_mcp_iccv \
    --num_runs=10 \
    --action_noise_std=0.1 \
    --pdp_policy_path=/home/mcarroll/Documents/cd-2/VideoMimic/PDP/outputs/2025.10.04/15.56.53_mdc_vidmimic/checkpoints/checkpoint_epoch_5.ckpt
    # --pdp_policy_path=/home/mcarroll/Documents/cd-2/VideoMimic/PDP/outputs/2025.09.28/21.05.52_pdp_phc_v2_vidmimic/checkpoints/checkpoint_epoch_35.ckpt

