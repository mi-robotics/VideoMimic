
cd PHC

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/

CUDA_LAUNCH_BLOCKING=1


# Base model dataset colelctions
# python scripts/phc_act/create_phc_act_dataset.py \
#     --dataset_path=data/amass/amass_train_upright.pkl \
#     --exp_name=phc_kp_mcp_iccv \
#     --num_runs=5 \
#     --action_noise_std=0.05

# Next step reference
# python scripts/phc_act/create_phc_act_dataset.py \
#     --dataset_path=data/amass/amass_train_upright.pkl \
#     --exp_name=phc_kp_mcp_iccv \
#     --num_runs=5 \
#     --action_noise_std=0.05


# HML3D dataset collections
# python scripts/phc_act/create_phc_act_dataset.py \
#     --dataset_path=data/amass/amass_train_upright_hml3d.pkl \
#     --exp_name=phc_kp_mcp_iccv \
#     --num_runs=5 \
#     --action_noise_std=0.05


# VidMimic dataset collections
# python scripts/phc_act/create_phc_act_dataset.py \
#     --dataset_path=data/amass/amass_train_upright_vidmimic.pkl \
#     --exp_name=phc_kp_mcp_iccv \
#     --num_runs=5 \
#     --action_noise_std=0.05


python scripts/phc_act/create_phc_act_dataset.py \
    --dataset_path=data/amass/amass_train_upright_run_jump_forward_backward.pkl \
    --exp_name=phc_kp_mcp_iccv \
    --num_runs=20 \
    --action_noise_std=0.05
