#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ===== 按你自己的实际模型路径改这里 =====
MODEL_PATH = "openai/gpt-oss-20b"

# ===== 按你机器的 GPU 改这里，比如只用 0,1,2,3 这四张卡做 TP=4 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


@dataclass
class VLLMEnv:
    llm: LLM
    tokenizer: AutoTokenizer


def build_vllm_env() -> VLLMEnv:
    """
    用你给的 reasoning_model_params 起一个 vLLM LLM 实例。
    """
    print(f"[INFO] Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"[INFO] Loading vLLM LLM from: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        tensor_parallel_size=4,          # reasoning_model_params["tensor_parallel_size"]
        dtype="bfloat16",                # reasoning_model_params["dtype"]
        max_num_seqs=64,                 # reasoning_model_params["max_num_seqs"]
        gpu_memory_utilization=0.80,     # reasoning_model_params["gpu_memory_utilization"]
        max_model_len=12288,             # reasoning_model_params["max_model_len"]
        max_num_batched_tokens=4096,     # reasoning_model_params["max_num_batched_tokens"]
        enable_prefix_caching=False,     # reasoning_model_params["enable_prefix_caching"]
        trust_remote_code=True,          # 视你的模型而定
    )

    return VLLMEnv(llm=llm, tokenizer=tokenizer)


def decode_new_tokens(env: VLLMEnv, prompt_token_ids: List[int], max_new_tokens: int = 256) -> str:
    """
    用 vLLM.generate 生成，只返回新生成的文本。
    max_new_tokens 可以比 reasoning_sampling_params["max_tokens"] 小一点，
    避免 debug 的时候输出太长。
    """

    # 你的 reasoning_sampling_params：
    # {
    #     "temperature": 0.4,
    #     "top_p": 0.95,
    #     "max_tokens": 8192,
    #     "repetition_penalty": 1.0,
    #     "presence_penalty": 0.0,
    #     "stop": ["<<<END>>>", "<|endoftext|>", "</s>", "<|im_end|>", "<|eot_id|>"]
    # }

    sp = SamplingParams(
        temperature=0.4,
        top_p=0.95,
        max_tokens=min(max_new_tokens, 8192),  # 这里用一个较小的上限更方便调试
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop=["<<<END>>>", "<|endoftext|>", "</s>", "<|im_end|>", "<|eot_id|>"],
    )

    outputs = env.llm.generate(
        prompt_token_ids=[prompt_token_ids],
        sampling_params=sp,
    )

    # vLLM: outputs[0].outputs[0].text 就是新生成内容
    out = outputs[0].outputs[0]
    return out.text

def main():
    env = build_vllm_env()
    tok = env.tokenizer

    chat = [
        {"role": "system", "content": "You are a mathematician. Solve the problem.\n\nReasoning: high"},
        {"role": "user", "content": "Solve the following problem:\nIn an acute scalene triangle $ABC$, points $D,E,F$ lie on sides $BC, CA, AB$, respectively, such that $AD \\perp BC, BE \\perp CA, CF \\perp AB$. Altitudes $AD, BE, CF$ meet at orthocenter $H$. Points $P$ and $Q$ lie on segment $EF$ such that $AP \\perp EF$ and $HQ \\perp EF$. Lines $DP$ and $QH$ intersect at point $R$. Compute $HQ/HR$."},
        {"role": "assistant", "content": "finalIn an acute scalene triangle \\(ABC\\), points \\(D, E, F\\) lie on sides \\(BC, CA, AB\\), respectively, such that \\(AD \\perp BC\\), \\(BE \\perp CA\\), \\(CF \\perp AB\\)."},
    ]

    # A: continue_final_message=True
    inputs_cont = tok.apply_chat_template(
        chat,
        tokenize=True,
        return_dict=True,
        continue_final_message=True,
        add_generation_prompt=False,
    )
    tpl = tok.chat_template
    print("chat template:", tpl)
    # apply_chat_template with return_dict=True returns a BatchEncoding or dict
    # but sometimes it returns a list of ints if return_dict is not respected or for single sequence
    # Let's handle both cases safely
    if isinstance(inputs_cont, list):
        ids_cont = inputs_cont
    elif hasattr(inputs_cont, "input_ids"):
        # If it's a BatchEncoding/dict, input_ids might be a tensor or list
        # If it's a list of lists (batch), take [0]. If it's a flat list, take it as is.
        # But apply_chat_template usually returns a flat list for single conversation unless return_tensors is set.
        # With return_dict=True, it returns {'input_ids': [...], 'attention_mask': [...]}
        ids_cont = inputs_cont["input_ids"]
    else:
        ids_cont = inputs_cont

    print(f"Prompt A (IDs): {ids_cont}")
    print(f"Prompt A (Decoded): {tok.decode(ids_cont)}")
    gen_cont = decode_new_tokens(env, ids_cont, max_new_tokens=4096)
    print(f"continue_final_message=True:{repr(gen_cont)}\n\n")

    # B: continue_final_message=False + add_generation_prompt=True
    inputs_new = tok.apply_chat_template(
        chat,
        tokenize=True,
        return_dict=True,
        continue_final_message=False,
        add_generation_prompt=True,
    )
    
    if isinstance(inputs_new, list):
        ids_new = inputs_new
    elif hasattr(inputs_new, "input_ids"):
        ids_new = inputs_new["input_ids"]
    else:
        ids_new = inputs_new

    print(f"Prompt B (IDs): {ids_new}")
    print(f"Prompt B (Decoded): {tok.decode(ids_new)}")
    gen_new = decode_new_tokens(env, ids_new, max_new_tokens=4096)
    print("continue_final_message=False:", repr(gen_new))

if __name__ == "__main__":
    main()