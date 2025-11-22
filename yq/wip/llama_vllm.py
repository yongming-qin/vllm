# 2025.10.22  
# Learning vllm/model_executor/llama.py  
# This file is recommended by Kuntai for learning.  


from vllm import LLM, SamplingParams


def main():
    llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)

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

    # llm.llm_engine.engine_core.shutdown()

    print(outputs[0].outputs[0].text)
    print(outputs[0].outputs[0].stop_reason)
    print(outputs[0].outputs[0].logprobs)

if __name__ == "__main__":
    main()
