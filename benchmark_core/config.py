from benchmark_core.paths import OMNI_MATH_DIR


Config = {
    "tag" : "1_29_0",
    "reasoning_model" : "Deepseek-v3.2",
    "reasoning_model_params": {
        "tensor_parallel_size": 4,          
        "dtype": "bfloat16",
        "max_num_seqs": 16,
        "gpu_memory_utilization": 0.75,    
        "max_model_len": 12288,             
        "max_num_batched_tokens": 1024,     
        "enable_prefix_caching": False,
    },
    "reasoning_sampling_params": {
        "temperature": 0.6,                 
        "top_p": 0.95,
        "max_tokens": 8192,
        "repetition_penalty": 1.05,
        "presence_penalty": 0.3,
        "frequency_penalty": 0.3,
        "stop": ["<<<END>>>", "<|endoftext|>", "</s>", "<|im_end|>", "<|eot_id|>"]
    },
    "reasoning_model_gpus" : "0,1,2,3",
    "judge_model_gpus" : "0,1,2,3",
    "judge_model" : "Qwen/Qwen3-32B",
    "judge_model_params" : {
        "tensor_parallel_size": 4,
        "dtype": "bfloat16",
        "max_num_seqs": 16,            
        "gpu_memory_utilization": 0.70
    },
    "judge_sampling_params" : {
        "temperature": 0,
        "top_p": 0.95,
        "max_tokens": 8192,
    }
    , 
    "Input_path" : str(OMNI_MATH_DIR / "Omni_MATH_Human_Segmented_100_1.jsonl"),
    "beta" : 1,
    "alpha" : 1,
    "lambda_h": 1,
    "threshold" : 0.6,
    "overall threshold" : 0.6,
    "max prefix_num" : 15,
    "skip_generate_num" : 0,
    "judge_aggregation" : "weighted",
    "judge_aggregation_weights": (0.4, 0.4, 0.2)
}
