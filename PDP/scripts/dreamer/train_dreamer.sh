HYDRA_FULL_ERROR=1 python train.py \
    --config-dir cfg/dreamer --config-name dreamer \
    logging.name=dreamer_test \
    hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \