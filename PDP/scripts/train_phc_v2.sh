HYDRA_FULL_ERROR=1 python train.py \
    --config-name=phc_v2.yaml \
    logging.name=pdp_phc_v2 \
    hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \