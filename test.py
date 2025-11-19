from vllm import LLM, SamplingParams

def main():
    # 一组示例 prompts
    prompts = [
        "介绍一下 Python，限制在一句话：",
        "把下面这句话翻译成英文：今天天气不错，就是有点冷。",
        "给我一个 1-10 的随机幸运数字，并解释为什么：",
    ]

    # 采样参数：这里 n=1，只要每个 prompt 一个结果
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=64,
        n=1,          # 每个 prompt 只要一个候选
    )

    # 创建 LLM 实例（模型名你按自己机器情况改）
    llm = LLM(model="facebook/opt-125m")

    # 直接把 list[str] 喂进去
    outputs = llm.generate(prompts, sampling_params)
    if isinstance(outputs, list):
        print("Outputs is a list.")
    
    # 逐个输出，对应关系是一一对应、按顺序不变
    for i, output in enumerate(outputs):
        prompt = output.prompt               # 原始 prompt
        generated = output.outputs[0].text   # 第一个候选的生成文本

        print("=" * 60)
        print(f"Prompt {i}: {prompt!r}\n")
        print("Generation:")
        print(generated.strip())
        print()

if __name__ == "__main__":
    main()
