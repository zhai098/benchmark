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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)



    def generate(self, prompt: str, schema: dict | None) -> str:
        ###后期增加统计tokens和延迟的功能
        sp = copy.deepcopy(self.sampling_params)
        if schema:
            sp.guided_decoding = GuidedDecodingParams(json=schema)
        else:
            sp.guided_decoding = None
            
        
        t0 = time.time()
        outs = self.llm.generate([prompt], sp)
        """model_inputs = self.tokenizer([prompt], return_tensors="pt")
        generated_ids = self.llm.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")"""
        latency = time.time() - t0

        text = outs[0].outputs[0].text.strip()
        print(f"[INFO] latency={latency:.3f}s")
        return text

        
    
        
