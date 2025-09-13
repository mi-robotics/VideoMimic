HYDRA_FULL_ERROR=1 python train.py \
    --config-name=phc.yaml \
    logging.name=pdp_phc \
    hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \