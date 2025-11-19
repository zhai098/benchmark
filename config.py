
Config = {
    "tag" : "long_output_11_16_0",
    "reasoning_model" : "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "reasoning_model_params" : {
        "tensor_parallel_size": 8,
        "dtype": "bfloat16",
        "max_num_seqs": 64,            
        "gpu_memory_utilization": 0.80,
        "enable_prefix_caching": False
    }
    ,
    "reasoning_sampling_params" : {
        "temperature": 0.4,
        "top_p": 0.95,
        "max_tokens": 8192,
        "repetition_penalty": 1.5,
        "stop": ["<<<END>>>"]       # Stop generation immediately once the sentinel appears
    },
    "reasoning_model_gpus" : "0,1,2,3,4,5,6,7",
    "judge_model_gpus" : "4,5,6,7",
    "judge_model" : "openai/gpt-oss-20b",
    "judge_model_params" : {
        "tensor_parallel_size": 4,
        "dtype": "bfloat16",
        "max_num_seqs": 16,            
        "gpu_memory_utilization": 0.80
    },
    "judge_sampling_params" : {
        "temperature": 0.3,
        "top_p": 0.95,
        "max_tokens": 4096
    }
    , 
    "Input_path" : "Omni_MATH/Omni_MATH_Human_Segmented.jsonl",
    "beta" : 1,
    "alpha" : 1,
    "lambda_h": 1,
    "threshold" : 0.6,
    "overall threshold" : 0.6,
    "max prefix_num" : 10,
    "skip_generate_num" : 0,
    "judge_aggregation" : "weighted",
    "judge_aggregation_weights": (0.4, 0.4, 0.2)
}