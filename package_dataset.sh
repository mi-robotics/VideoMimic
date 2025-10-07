cd PHC

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/

# # Base model dataset collections
# python scripts/phc_act/package_ds.py \
#     --dataset_path=data/amass/amass_train_upright.pkl \
#     --exp_name=phc_kp_mcp_iccv \
#     --num_runs=10 \
#     --action_noise_std=0.1


# HML3D dataset collections
# python scripts/phc_act/package_ds.py \
#     --dataset_path=data/amass/amass_train_upright_with_ref_1.pkl \
#     --exp_name=phc_kp_mcp_iccv \
#     --num_runs=10 \
#     --action_noise_std=0.1

# Vidmimic dataset collections
python scripts/phc_act/package_ds.py \
    --dataset_path=data/amass/amass_train_upright.pkl \
    --exp_name=phc_kp_mcp_iccv \
    --num_runs=10 \
    --action_noise_std=0.1