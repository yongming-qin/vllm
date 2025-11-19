# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Lazy-loaded model config registry for vLLM.

This module lazily exposes config classes without importing all
config submodules upfront, dramatically reducing import time.

Behavior:
- `configs.<ConfigClass>` triggers a dynamic import of only the
  corresponding module.
- Keeps full compatibility with existing vLLM imports.
"""

from __future__ import annotations
import importlib
from typing import Dict


# ------------------------------------------------------------------------------
# List of all exportable config classes.
# These are *names only*; they are NOT imported yet.
# ------------------------------------------------------------------------------
__all__ = [
    "AfmoeConfig",
    "ChatGLMConfig",
    "DeepseekVLV2Config",
    "DeepseekV3Config",
    "DotsOCRConfig",
    "EAGLEConfig",
    "FlexOlmoConfig",
    "RWConfig",
    "JAISConfig",
    "Lfm2MoeConfig",
    "MedusaConfig",
    "MiDashengLMConfig",
    "MLPSpeculatorConfig",
    "MoonViTConfig",
    "KimiLinearConfig",
    "KimiVLConfig",
    "NemotronConfig",
    "NemotronHConfig",
    "Olmo3Config",
    "OvisConfig",
    "RadioConfig",
    "SpeculatorsConfig",
    "UltravoxConfig",
    "Step3VLConfig",
    "Step3VisionEncoderConfig",
    "Step3TextConfig",
    "Qwen3NextConfig",
]


# ------------------------------------------------------------------------------
# Map class names → module paths.
# Only this dictionary is evaluated at import time.
# Each class is imported lazily when first accessed.
# ------------------------------------------------------------------------------
_CLASS_TO_MODULE: Dict[str, str] = {
    "AfmoeConfig": "vllm.transformers_utils.configs.afmoe",
    "ChatGLMConfig": "vllm.transformers_utils.configs.chatglm",
    "DeepseekVLV2Config": "vllm.transformers_utils.configs.deepseek_vl2",
    "DotsOCRConfig": "vllm.transformers_utils.configs.dotsocr",
    "EAGLEConfig": "vllm.transformers_utils.configs.eagle",
    "FlexOlmoConfig": "vllm.transformers_utils.configs.flex_olmo",
    "RWConfig": "vllm.transformers_utils.configs.falcon",
    "JAISConfig": "vllm.transformers_utils.configs.jais",
    "Lfm2MoeConfig": "vllm.transformers_utils.configs.lfm2_moe",
    "MedusaConfig": "vllm.transformers_utils.configs.medusa",
    "MiDashengLMConfig": "vllm.transformers_utils.configs.midashenglm",
    "MLPSpeculatorConfig": "vllm.transformers_utils.configs.mlp_speculator",
    "MoonViTConfig": "vllm.transformers_utils.configs.moonvit",
    "KimiLinearConfig": "vllm.transformers_utils.configs.kimi_linear",
    "KimiVLConfig": "vllm.transformers_utils.configs.kimi_vl",
    "NemotronConfig": "vllm.transformers_utils.configs.nemotron",
    "NemotronHConfig": "vllm.transformers_utils.configs.nemotron_h",
    "Olmo3Config": "vllm.transformers_utils.configs.olmo3",
    "OvisConfig": "vllm.transformers_utils.configs.ovis",
    "RadioConfig": "vllm.transformers_utils.configs.radio",
    "SpeculatorsConfig": "vllm.transformers_utils.configs.speculators.base",
    "UltravoxConfig": "vllm.transformers_utils.configs.ultravox",
    "Step3VLConfig": "vllm.transformers_utils.configs.step3_vl",
    "Step3VisionEncoderConfig": "vllm.transformers_utils.configs.step3_vl",
    "Step3TextConfig": "vllm.transformers_utils.configs.step3_vl",
    "Qwen3NextConfig": "vllm.transformers_utils.configs.qwen3_next",

    # Special case: DeepseekV3Config is from HuggingFace Transformers
    "DeepseekV3Config": "transformers",
}


# ------------------------------------------------------------------------------
# Lazy attribute loader (PEP 562)
# ------------------------------------------------------------------------------
def __getattr__(name: str):
    """Lazily load config classes when accessed.

    Example:
        from vllm.transformers_utils.configs import DeepseekVLV2Config
        # This triggers a dynamic import of configs.deepseek_vl2
    """
    if name in _CLASS_TO_MODULE:
        module_name = _CLASS_TO_MODULE[name]
        module = importlib.import_module(module_name)
        return getattr(module, name)

    raise AttributeError(f"module 'configs' has no attribute '{name}'")


# ------------------------------------------------------------------------------
# Optional: Improve autocomplete in IDEs
# Create module attributes by wrapping __getattr__
# ------------------------------------------------------------------------------
def __dir__():
    return sorted(list(__all__))
