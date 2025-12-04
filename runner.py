from openai import OpenAI
import os
import time
import copy


class VLLMRunner:
    def __init__(self, model: str, vllm_config: dict, sampling_config: dict, gpus: str):
        # 这些信息主要用于记录 & 方便你在别处使用
        self.model_name = model
        self.vllm_config = vllm_config
        self.sampling_config = sampling_config
        self.gpus = gpus

        # openai 客户端，指向 vLLM 的 OpenAI 兼容服务
        self.client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",          # 和 vLLM 启动脚本里保持一致
        )

        # 采样配置直接缓存，用于每次请求时传给 vLLM
        self.temperature = sampling_config.get("temperature", 0.7)
        self.top_p = sampling_config.get("top_p", 0.95)
        self.max_tokens = sampling_config.get("max_tokens", 256)
        self.repetition_penalty = sampling_config.get("repetition_penalty", 1.0)
        self.presence_penalty = sampling_config.get("presence_penalty", 0.0)
        self.stop = sampling_config.get("stop", None)
        self.debug_prompt = ""

    def generate(self, prompt, schema=None) -> list[str]:
        """
        支持三种输入：
        1) str
        2) list[dict]（单个 messages）
        3) list[list[dict]]（多个 messages 批处理）
        """

        # --- 情况 3: 批量 list[list[dict]] ---
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], list):
            results = []
            for idx, one_messages in enumerate(prompt):
                print(f"[DEBUG] Generating batch item {idx+1}/{len(prompt)}")
                if not (isinstance(one_messages, list) and isinstance(one_messages[0], dict)):
                    raise TypeError("Batch item must be list[dict] messages")

                out = self.generate(one_messages, schema)   # 递归调用单条生成
                results.extend(out)
            return results

        # --- 情况 2: 单条 list[dict] ---
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
            messages = prompt


        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        # --- 调用 openai 接口 ---
        extra_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
        }

        if schema is not None:
            extra_params["guided_json"] = schema

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **extra_params
        )

        text = completion.choices[0].message.content
        print(completion)
        return [text]
