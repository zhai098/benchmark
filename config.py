

Config = {
    "reasoning_model" : "Qwen/Qwen3-0.6B",
    "reasoning_model_params" : {
        "tensor_parallel_size": 4,
        "dtype": "bfloat16",
        "max_num_seqs": 16,            
        "gpu_memory_utilization": 0.80 
    }
    ,
    "reasoning_sampling_params" : {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096,
        "stop": ["<<<END>>>"]       # 命中哨兵立即停
    },
    "reasoning_model_gpus" : "0,1,2,3",
    "judge_model_gpus" : "4,5,6,7",
    "judge_model" : "Qwen/Qwen3-0.6B",
    "judge_model_params" : {
        "tensor_parallel_size": 4,
        "dtype": "bfloat16",
        "max_num_seqs": 16,            
        "gpu_memory_utilization": 0.80 
    },
    "judge_sampling_params" : {
        "temperature": 0.1,
        "top_p": 0.95,
        "max_tokens": 1024
    }
    ,
    "Input_path" : "Omni_MATH/Omni_MATH_Long_Segmented.jsonl",
    "beta" : 1,
    "alpha" : 1,
    "lambda_h": 1,
    "threshold" : 0.8,
    "overall threshold" : 0.6,
    "max prefix_num" : 10,
}