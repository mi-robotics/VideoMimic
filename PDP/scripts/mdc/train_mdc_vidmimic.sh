HYDRA_FULL_ERROR=1 python train.py \
    --config-name=mdc_vidmimic.yaml \
    logging.name=mdc_vidmimic \
    hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \