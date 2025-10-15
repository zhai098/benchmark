

Config = {
    "reasoning_model" : "mistralai/Mistral-7B-Instruct-v0.2",
    "reasoning_model_params" : {
        "tensor_parallel_size": 2,
        "dtype": "bfloat16",
        "max_num_seqs": 16,            
        "gpu_memory_utilization": 0.80 
    }
    ,
    "reasoning_sampling_params" : {
        "temperature": 0.4,
        "top_p": 0.95,
        "max_tokens": 2048,
        "repetition_penalty": 1.5,
        "stop": ["<<<END>>>"]       # Stop generation immediately once the sentinel appears
    },
    "reasoning_model_gpus" : "8,9",
    "judge_model_gpus" : "4,5,6,7",
    "judge_model" : "Qwen/Qwen3-8B",
    "judge_model_params" : {
        "tensor_parallel_size": 4,
        "dtype": "bfloat16",
        "max_num_seqs": 16,            
        "gpu_memory_utilization": 0.80 
    },
    "judge_sampling_params" : {
        "temperature": 0.3,
        "top_p": 0.95,
        "max_tokens": 2048
    }
    , 
    "Input_path" : "Omni_MATH/Omni_MATH_Long_Segmented.jsonl",
    "beta" : 1,
    "alpha" : 1,
    "lambda_h": 1,
    "threshold" : 0.6,
    "overall threshold" : 0.6,
    "max prefix_num" : 10,
}