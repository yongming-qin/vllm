"""
Test prismatic vision backbone loading and generation with vllm.

2025.11.03

"""


import torch
from vllm.model_executor.models.openvla import PrismaticVisionBackbone
from vllm.config import ModelConfig, VllmConfig
from transformers import AutoConfig

config = AutoConfig.from_pretrained('/home/yq/ssd/vllm-dir/vllm/yq/openvla-7b', trust_remote_code=True)
vllm_config = VllmConfig(model='/home/yq/ssd/vllm-dir/vllm/yq/openvla-7b', trust_remote_code=True)
backbone = PrismaticVisionBackbone(config, None, prefix='vision_backbone')
print('Featurizer params (first 5):')
for i, (name, _) in enumerate(backbone.featurizer.named_parameters()):
    if i < 5:
        print(f'  {name}')
    else:
        break
print('Fused featurizer params (first 5):')
for i, (name, _) in enumerate(backbone.fused_featurizer.named_parameters()):
    if i < 5:
        print(f'  {name}')
    else:
        break