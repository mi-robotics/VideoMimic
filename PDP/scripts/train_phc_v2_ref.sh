HYDRA_FULL_ERROR=1 python train.py \
    --config-name=phc_v2_ref.yaml \
    logging.name=pdp_phc_v2_ref \
    hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \