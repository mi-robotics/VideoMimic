HYDRA_FULL_ERROR=1 python train.py \
    --config-name=mdc_pref_comp.yaml \
    logging.name=mdc_pref_comp \
    hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \