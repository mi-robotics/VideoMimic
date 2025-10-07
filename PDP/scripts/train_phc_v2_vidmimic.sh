HYDRA_FULL_ERROR=1 python train.py \
    --config-name=phc_v2_vidmimic.yaml \
    logging.name=pdp_phc_v2_vidmimic \
    hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \