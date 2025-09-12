
cd PHC

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/

python scripts/phc_act/create_phc_act_dataset.py \
    --dataset_path=data/amass/amass_train_upright.pkl \
    --exp_name=phc_kp_mcp_iccv \
    --num_runs=10 \
    --action_noise_std=0.1