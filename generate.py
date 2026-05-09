# stage1_generate.py
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Sequence, Tuple
from benchmark_core.config import Config
from benchmark_core.prompt import Generate_Prompt
from benchmark_core.data_process import Processor, _write_jsonl_line, _write_pretty_json, _normalize_generation_input

logger = logging.getLogger(__name__)
processor = Processor()

def build_reasoning_model():
    from runner import VLLMRunner

    return VLLMRunner(
        model=Config["reasoning_model"],
        vllm_config=Config["reasoning_model_params"],
        sampling_config=Config["reasoning_sampling_params"],
        gpus=Config["reasoning_model_gpus"],
    )


def _split_generate_response(response: Any) -> Tuple[List[str], List[str]]:
    def _as_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item or "") for item in value]

    if isinstance(response, tuple) and len(response) == 2:
        reasonings, generations = response
        generations_list = _as_list(generations)
        reasonings_list = _as_list(reasonings)
        if len(reasonings_list) < len(generations_list):
            reasonings_list.extend([""] * (len(generations_list) - len(reasonings_list)))
        return reasonings_list[: len(generations_list)], generations_list
    generations = _as_list(response)
    return [""] * len(generations), generations


def _call_reasoning_model(reasoning_model: Any, prompts: Sequence[Any]) -> Any:
    return reasoning_model.generate(list(prompts), None)


def _generate_once(
    reasoning_model: Any,
    prompts: Sequence[Any],
) -> Tuple[List[str], List[str], List[int]]:
    reasonings, generations = _split_generate_response(_call_reasoning_model(reasoning_model, prompts))
    assert len(generations) == len(prompts), (
        "Reasoning model returned a different number of generations than prompts: "
        f"prompts={len(prompts)}, generations={len(generations)}. "
        "Runner implementations must preserve one output slot per prompt; "
        "do not drop failed middle items."
    )
    assert len(reasonings) == len(prompts), (
        "Reasoning model returned a different number of reasonings than prompts: "
        f"prompts={len(prompts)}, reasonings={len(reasonings)}."
    )

    final_empty_indices = [
        idx
        for idx, text in enumerate(generations)
        if not _normalize_generation_input(text).strip()
    ]
    if final_empty_indices:
        print(f"[ERROR][GENERATE] empty generation outputs at indices {final_empty_indices}")
    return reasonings, generations, final_empty_indices


def _parse_legacy_step_payload(value: Any) -> Dict[str, Any] | None:
    """Decode legacy stringified {'id': ..., 'text': ...} step payloads."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not (text.startswith("{") and text.endswith("}") and "text" in text):
        return None
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _clean_reference_step_text(value: Any) -> str:
    parsed = _parse_legacy_step_payload(value)
    if parsed is not None:
        value = parsed.get("text") or parsed.get("content") or ""
    return str(value or "").strip()


def _reference_step_records(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "reference_steps" in obj:
        raw_steps = obj.get("reference_steps") or []
    elif "steps" in obj:
        raw_steps = obj.get("steps") or []
    else:
        raw_steps = obj.get("segments") or []
    records: List[Dict[str, Any]] = []
    for idx, step in enumerate(raw_steps):
        if isinstance(step, dict):
            text = _clean_reference_step_text(step.get("text") or step.get("content") or "")
            step_type = str(step.get("type") or "text")
        else:
            text = _clean_reference_step_text(step)
            step_type = "text"
        if not text:
            continue
        records.append({"text": text, "type": step_type, "index": idx})
    return records


def _reference_steps_for_output(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = _reference_step_records(obj)
    return [
        {
            "step_id": f"s{idx + 1}",
            "text": record["text"],
            "type": record["type"],
        }
        for idx, record in enumerate(records)
    ]


_SPECIAL_GENERATION_TOKENS = (
    "<|tool_call_end|>",
    "<|tool_call_start|>",
    "<|tool▁call▁end|>",
    "<|tool▁call▁start|>",
)


def _clean_generation_output(value: Any) -> str:
    text = _normalize_generation_input(value)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.replace("<think>", "")
    for token in _SPECIAL_GENERATION_TOKENS:
        text = text.replace(token, " ")
    return re.sub(r"\s+", " ", text).strip()


def _reference_claims_by_step(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    return obj.get("reference_claims_by_step") or obj.get("claims_by_step") or []


def _reference_step_dependencies(obj: Dict[str, Any]) -> Dict[str, Any]:
    return obj.get("reference_step_dependencies") or obj.get("step_dependencies") or {}


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if isinstance(value, str):
            if value.strip():
                return value
        elif value is not None:
            return value
    return None


def _record_identity(obj: Dict[str, Any], case_id: Any, row_num: int) -> Tuple[str, str]:
    annotation_uid = _first_nonempty(obj.get("annotation_uid"), obj.get("uid"), obj.get("uuid"))
    if annotation_uid:
        text = str(annotation_uid)
        return text, text

    raw_id = _first_nonempty(obj.get("id"))
    sample_idx = obj.get("sample_idx")
    annotator_id = obj.get("annotator_id")
    device_id = obj.get("device_id")
    case_text = str(case_id)

    if sample_idx is not None:
        parts = [case_text]
        if annotator_id:
            parts.append(str(annotator_id))
        if device_id:
            parts.append(str(device_id))
        parts.append(f"sample_{sample_idx}")
        text = "__".join(parts)
        return text, text

    if raw_id and str(raw_id) != case_text:
        text = str(raw_id)
        return text, text

    text = f"{case_text}__row_{row_num:06d}"
    return text, text

def generate_case(obj: Dict[str, Any], reasoning_model) -> Dict[str, Any]:
    """复用 main.py 的生成阶段：逐步 add_step -> run()，直到用尽参考步骤。
       逻辑等价于 execute_evaluation() 中生成部分。"""
    problem = obj.get("question") or obj["problem"]
    thought_seg = _reference_step_records(obj)
    answer = obj.get("standard_solution") or obj.get("solution") or obj.get("answer", "")

    # 初始化 prompt（与原逻辑一致）
    generate_promptbuilder = Generate_Prompt(reasoning_model, query=problem)

    # 只取 content，等价于你现在的 thought_policy 构造
    thought_policy: List[str] = [seg["text"] for seg in thought_seg]
    type_list: List[str] = [seg["type"] for seg in thought_seg]
    unprocessed: List[Tuple[str, str]] = list(zip(thought_policy, type_list))
    processed: List[str] = []
    processed_types: List[str] = []
    gen_output: List[str] = []
    gen_prefixes: List[str] = []
    i = 1
    prompt_lists = []
    while unprocessed:
        current_step, current_type = unprocessed.pop(0)
        #print(f"[DEBUG] Generating step {i}, step={current_step}")
        generate_promptbuilder.add_step(current_step)
                
        # Use IDs for generation if available
        if hasattr(generate_promptbuilder, "return_prompt_ids"):
            prompt_lists.append(generate_promptbuilder.return_prompt_ids())
        else:
            prompt_lists.append(generate_promptbuilder.return_prompt())

        processed.append(current_step)
        processed_types.append(current_type)

        if not unprocessed:
            print("[DEBUG] Reached last step (generation), stop generation loop")
            break
        i += 1
    # 一次性生成所有步骤
    prompts = prompt_lists 
    reasonings, generations, empty_generation_indices = _generate_once(reasoning_model, prompts)
    cleaned_generations = [_clean_generation_output(gen) for gen in generations]
    gen_output.extend(cleaned_generations)
    for current_output in cleaned_generations:
        gen_sents_all = processor.sentence_split_en(current_output)
        K = max(1, min(Config["max prefix_num"], len(gen_sents_all)))
        gen_sents = gen_sents_all[:K]
        gen_prefix = " ".join(gen_sents)
        gen_prefixes.append(gen_prefix)
        #print(f"[DEBUG] Final generated prefix: {gen_prefix}\n")
        #print(f"[DEBUG] Full generated output: {current_output}\n")
    
    return {
        "problem": problem,
        "answer": answer,
        "prompts": prompts,       # 复现用
        "ref_steps": processed,     # 评测阶段需要参考步骤（与 processed_thought 等价）
        "gen_output": gen_output,   # 待评测的模型生成
        "gen_prefix": gen_prefixes,
        "reasoning": reasonings,
        "empty_generation_indices": empty_generation_indices,
        "empty_generation_count": len(empty_generation_indices),
        "difficulty": float(obj.get("difficulty", 0.0)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=Config["Input_path"])
    parser.add_argument("--out_root", type=str, default=None)
    parser.add_argument("--tag", type=str, default=Config["tag"])
    parser.add_argument("--max_cases", type=int, default=100)
    parser.add_argument(
        "--use_vllm_local",
        action="store_true",
        help="Deprecated no-op; generate.py now always uses local vLLM.",
    )
    args = parser.parse_args()

    # 输出目录命名沿用 main.py 风格
    out_root = os.path.abspath(args.out_root or "./gen_output")
    os.makedirs(out_root, exist_ok=True)

    tag = args.tag
    model_label = os.path.basename(os.path.normpath(str(Config["reasoning_model"]))) or "model"
    run_dir_name = f"{model_label}_{tag}"
    run_dir = os.path.join(out_root, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    # 只生成、不评测：生成专用文件名
    gen_only_jsonl = os.path.join(run_dir, "gen_only.jsonl")
    gen_only_pretty = os.path.join(run_dir, "gen_only_pretty.json")
    manifest_path = os.path.join(run_dir, "run_info.json")

    input_path = args.input_path
    print(f"[INFO][GENERATE] loading: {input_path}")
    reasoning_model = build_reasoning_model()
    num = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
        open(gen_only_jsonl, "w", encoding="utf-8", buffering=1) as fgen, \
        open(gen_only_pretty, "w", encoding="utf-8", buffering=1) as fgen_pretty:

        for line in fin:
            if num >= args.max_cases:
                break
            if num < Config.get("skip_generate_num", 0):
                num += 1
                continue
            t0 = time.time()
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("[WARN] skip one bad json line")
                continue
            num += 1

            res = generate_case(obj, reasoning_model)
            output_steps = _reference_steps_for_output(obj)
            case_id = _first_nonempty(obj.get("case_id"), obj.get("original_case_id"), f"q-{num}")
            record_id, annotation_uid = _record_identity(obj, case_id, num)

            # 携带 id，方便第二阶段对齐
            out_record = {
                "id": record_id,
                "case_id": case_id,
                "annotation_uid": annotation_uid,
                "original_case_id": _first_nonempty(obj.get("original_case_id"), case_id),
                "annotator_id": obj.get("annotator_id"),
                "device_id": obj.get("device_id"),
                "sample_idx": obj.get("sample_idx"),
                "detail_path": obj.get("detail_path"),
                "difficulty": res["difficulty"],
                "question": res["problem"],
                "problem": res["problem"],
                "standard_solution": obj.get("standard_solution", ""),
                "answer": res["answer"],
                "prompts": res["prompts"],        # 复现用
                "ref_steps": res["ref_steps"],      # 评测要用
                "gen_output": res["gen_output"],    # 评测要用
                "gen_prefix": res["gen_prefix"],    # 评测要用
                "reasoning_content": res["reasoning"],
                "empty_generation_indices": res["empty_generation_indices"],
                "empty_generation_count": res["empty_generation_count"],
                "has_correct_sample": obj.get("has_correct_sample", False),
                "correct_sample_idx": obj.get("correct_sample_idx"),
                "correct_sample_solution": obj.get("correct_sample_solution", ""),
                "steps": output_steps,
                "claims_by_step": _reference_claims_by_step(obj),
                "step_dependencies": _reference_step_dependencies(obj),
            }
            _write_jsonl_line(fgen, out_record)
            _write_pretty_json(fgen_pretty, out_record)

            t1 = time.time()
            print(f"[INFO][GENERATE] generated case {num}, time={t1 - t0:.2f}s")

    # 写一个小 manifest，第二阶段方便指向文件
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_dir": run_dir,
            "gen_file": gen_only_jsonl,
            "model_reasoning": Config["reasoning_model"],
            "model_judge": Config["judge_model"],
            "num": num,
            "backend": "vllm",
            "use_vllm_local": True,
        }, f, ensure_ascii=False, indent=2)

    print(f"[RESULT][GENERATE] wrote {num} generations to: {gen_only_jsonl}")
    print(f"[RESULT][GENERATE] run_dir = {run_dir}")

if __name__ == "__main__":
    main()
