"""
Use an optimized vllm model to generate text.
Will use model_impl='transformers' to load the model.
Thus, I can learn from the code for template.

Yongming
2025.10.23
"""


from vllm import LLM, SamplingParams
import logging
logging.basicConfig(level=logging.INFO)

# checked the config.json.
# facebook/opt-125m architecture is OPTForCausalLM
# which is optimized in vllm.

# same for sshleifer/tiny-gpt2, architecture is GPT2LMHeadModel
# same for distilgpt2, architecture is GPT2LMHeadModel


#yq facebook/opt-125m, Qwen/Qwen2-VL-2B-Instruct don't release
# the modeling.py and configuration.py in their hf repo.
# transformers implement the model in the source code.

llm = LLM(
    # model="facebook/opt-125m",
    model="/home/yq/ssd/vllm-dir/vllm/examples/yq/opt-125m",
    # model="Qwen/Qwen2-VL-2B-Instruct",
    model_impl='transformers',
    trust_remote_code=True,
    dtype='bfloat16',
    runner='generate'
)
print("=================")
print(llm.model_config.model_impl)
print("=================")

import sys; sys.exit()

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,
)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

outputs = llm.generate(prompts, sampling_params)

print(outputs[0].outputs[0].text)
print(outputs[0].outputs[0].stop_reason)
print(outputs[0].outputs[0].logprobs)


llm.llm_engine.engine_core.shutdown()

import sys; sys.exit()