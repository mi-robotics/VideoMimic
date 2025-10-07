HYDRA_FULL_ERROR=1 python train.py \
    --config-name=phc_v2_hml3d.yaml \
    logging.name=pdp_phc_v2_hml3d_ft \
    hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \