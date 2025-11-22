

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def vllm_gpt2():
    model = LLM(
        model="/home/yq/ssd/vllm-dir/vllm/yq/gpt2",
        tokenizer="gpt2",
        dtype="float32",
        model_impl="vllm",
    )

    prompt = "Hello, my name is"

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=30
    )

    outputs = model.generate([prompt], sampling_params)

    print(outputs[0].outputs[0].text)
    

def transformers_gpt2():
    config = AutoConfig.from_pretrained("/home/yq/ssd/vllm-dir/vllm/yq/gpt2")
    print(config)
    
    import ipdb; ipdb.set_trace()
    
    tokenizer = AutoTokenizer.from_pretrained("/home/yq/ssd/vllm-dir/vllm/yq/gpt2")
    model = AutoModelForCausalLM.from_pretrained("/home/yq/ssd/vllm-dir/vllm/yq/gpt2")
    prompt = "Hello, my name is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print(inputs)
    # import ipdb; ipdb.set_trace()
    
    outputs = model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    
if __name__ == "__main__":
    # vllm_gpt2()
    transformers_gpt2()