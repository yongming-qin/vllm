"""
Looking for a hf repo model that is not optimized for vllm.
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

llm = LLM(
    model="EleutherAI/gpt-neo-125M",
    model_impl='transformers',
    trust_remote_code=True,
    dtype='bfloat16',
    runner='generate'
)
print(llm.model_config.model_type)

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

