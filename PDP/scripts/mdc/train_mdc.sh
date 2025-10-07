HYDRA_FULL_ERROR=1 python train.py \
    --config-name=mdc_base.yaml \
    logging.name=mdc_base_40rjfb_causalMask \
    hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \