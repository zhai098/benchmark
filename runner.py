from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
import os
import time
import copy 
class VLLMRunner:
    def __init__(self, model: str, vllm_config: dict, sampling_config: dict, gpus: str):
        self.model_name = model
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        self.llm = LLM(model=model,
            tokenizer=model,
            **vllm_config)
        self.sampling_params = SamplingParams(temperature=sampling_config.get("temperature", 0.7),
            top_p=sampling_config.get("top_p", 0.95),
            max_tokens=sampling_config.get("max_tokens", 256),
            stop=sampling_config.get("stop", ["<<<END>>>"]))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)



    def generate(self, prompt: str | list[str] | list[int] | list[list[int]], schema: dict | None) -> list[str]:
        ###后期增加统计tokens和延迟的功能
        sp = copy.deepcopy(self.sampling_params)
        if schema:
            sp.guided_decoding = GuidedDecodingParams(json=schema)
        else:
            sp.guided_decoding = None
            
        t0 = time.time()
        
        # Check if prompt is token IDs (list of ints or list of list of ints)
        is_token_ids = False
        if isinstance(prompt, list):
            if len(prompt) > 0:
                if isinstance(prompt[0], int):
                    # Single prompt as list of ints
                    is_token_ids = True
                    prompt_arg = None
                    prompt_token_ids = [prompt]
                elif isinstance(prompt[0], list) and len(prompt[0]) > 0 and isinstance(prompt[0][0], int):
                    # Batch of prompts as list of list of ints
                    is_token_ids = True
                    prompt_arg = None
                    prompt_token_ids = prompt
                else:
                    # List of strings
                    prompt_arg = prompt
                    prompt_token_ids = None
            else:
                # Empty list
                prompt_arg = prompt
                prompt_token_ids = None
        else:
            # Single string
            prompt_arg = [prompt]
            prompt_token_ids = None

        if is_token_ids:
            print("Generating for a list of token IDs, count:", len(prompt_token_ids))
            outs = self.llm.generate(prompts=None, sampling_params=sp, prompt_token_ids=prompt_token_ids)
        else:
            if isinstance(prompt, list):
                print("Generating for a list of prompts, count:", len(prompt))
                outs = self.llm.generate(prompt, sp)
            else:
                outs = self.llm.generate([prompt], sp)
        
        latency = time.time() - t0

        texts = []
        for i, out in enumerate(outs):
            text = out.outputs[0].text
            texts.append(text)
        
        print(f"[INFO] latency={latency:.3f}s")
        return texts

        
    
        
