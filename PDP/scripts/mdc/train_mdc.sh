# HYDRA_FULL_ERROR=1 python train.py \
#     --config-name=mdc_base.yaml \
#     logging.name=mdc_base_40rjfb_causalMask \
#     hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \


HYDRA_FULL_ERROR=1 python train.py \
    --config-name=mdc_base.yaml \
    logging.name=mdc_base_40rjfb_ClocMask \
    policy.model.causal_attn_type=cloc \
    hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \


HYDRA_FULL_ERROR=1 python train.py \
    --config-name=mdc_base.yaml \
    logging.name=mdc_base_40rjfb_Cloc+Mask \
    policy.model.causal_attn_type=cloc+ \
    hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \

HYDRA_FULL_ERROR=1 python train.py \
    --config-name=mdc_base.yaml \
    logging.name=mdc_base_40rjfb_NoMask \
    policy.model.causal_attn=False \
    policy.model.causal_attn_type=none \
    hydra.run.dir=outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${logging.name} \