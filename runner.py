from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import copy
from typing import List, Union, Dict, Any, Tuple
import math
from openai import OpenAI
from config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, threading
import httpx
import sys
import asyncio
import json
import inspect
import torch
from typing import Any, Dict, List, Tuple, Union
from volcenginesdkarkruntime import AsyncArk

class VLLMRunner:
    def __init__(self, model: str, vllm_config: dict, sampling_config: dict, gpus: str):
        self.model_name = model
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        # Some Hugging Face model repos require executing custom code.
        # Allow it by default for this benchmark, but keep it overridable.
        vllm_kwargs = dict(vllm_config or {})
        vllm_kwargs.setdefault("trust_remote_code", True)
        self.llm = LLM(
            model=model,
            tokenizer=model,
            **vllm_kwargs,
        )
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

    @staticmethod
    def _get_lp_value(lp_obj: Any) -> float | None:
        """vLLM logprob object compatibility: sometimes has .logprob, sometimes is float."""
        if lp_obj is None:
            return None
        return float(getattr(lp_obj, "logprob", lp_obj))

    def score(
        self,
        problem: str,
        solution: str,
        sep: str = "\n\n",
        *,
        prompt_logprobs_k: int = 1,
    ) -> dict:
        """
        Return log-likelihood metrics of `solution` given `problem` under THIS vLLM model.

        Computes sum log p(token) over solution tokens only:
        logp = Σ log p(y_t | problem, y_<t)
        plus avg_nll and ppl.

        Keep problem/sep constant across candidates for fair comparison.
        """

        system_message = (
            "You are a mathematician. Solve the problem."
            "## Style preferences (keep them light; do not change your underlying approach):"
                "- Treat `current_solution`/`ref` as correct established premises and build directly on them."
                "- Start immediately with the next logical derivation. Do not restate the problem or re-summarize what has already been established."
                "- Write as continuous mathematical prose (no section headers, no “Step 1/2/3”)."
                "- Avoid repeating the same conditions. If you must reference a prior premise, do it minimally (e.g., “from the previous inequality …”)."
        )
        user_message = f"Solve the Problem:\n{problem}"
        prefix_message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        prefix_prompt = self.tokenizer.apply_chat_template(
            prefix_message,
            tokenize=False,
            enable_thinking=True,
        )
        full_message = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": solution},
        ]
        #print(full_message)
        prompt = self.tokenizer.apply_chat_template(
            full_message,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
            enable_thinking=True,
        )
        boundary = len(self.tokenizer.encode(prefix_prompt, add_special_tokens=False))

        sp = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,                 # generate 1 token, ignore it; we only use prompt_logprobs
            prompt_logprobs=prompt_logprobs_k,
        )

        out = self.llm.generate([prompt], sp)[0]
        prompt_ids = out.prompt_token_ids
        prompt_lps = out.prompt_logprobs  # list[Optional[dict[token_id -> logprob]]]
        print(prompt_lps)
        if prompt_lps is None:
            raise RuntimeError("prompt_logprobs not returned. Check vLLM version / SamplingParams(prompt_logprobs=...).")

        if boundary > len(prompt_ids):
            raise RuntimeError(f"Boundary token index {boundary} > prompt length {len(prompt_ids)}. Check sep/prefix.")

        logp = 0.0
        n = 0
        for i in range(boundary, len(prompt_ids)):
            d = prompt_lps[i]
            if not d:
                continue
            tid = prompt_ids[i]
            lp = self._get_lp_value(d.get(tid))
            if lp is None:
                continue
            logp += lp
            n += 1

        avg_nll = -logp / max(n, 1)
        ppl = math.exp(avg_nll)

        return {
            "logp_answer_given_problem": logp,
            "answer_tokens_counted": n,
            "avg_nll": avg_nll,
            "ppl": ppl,
        }

    def score_batch(
        self,
        problems: list[str],
        answers: list[str],
        sep: str = "\n\n",
        *,
        prompt_logprobs_k: int = 1,
    ) -> list[dict]:
        """Batch version: same computation, one vLLM call."""
        assert len(problems) == len(answers)
        prefixes = [p + sep for p in problems]
        fulls = [pref + a for pref, a in zip(prefixes, answers)]
        boundaries = [len(self.tokenizer.encode(pref, add_special_tokens=False)) for pref in prefixes]

        sp = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            prompt_logprobs=prompt_logprobs_k,
        )

        outs = self.llm.generate(fulls, sp)

        results = []
        for out, boundary in zip(outs, boundaries):
            prompt_ids = out.prompt_token_ids
            prompt_lps = out.prompt_logprobs
            if prompt_lps is None:
                raise RuntimeError("prompt_logprobs not returned. Check vLLM version / SamplingParams(prompt_logprobs=...).")
            if boundary > len(prompt_ids):
                raise RuntimeError(f"Boundary token index {boundary} > prompt length {len(prompt_ids)}.")

            logp = 0.0
            n = 0
            for i in range(boundary, len(prompt_ids)):
                d = prompt_lps[i]
                if not d:
                    continue
                tid = prompt_ids[i]
                lp = self._get_lp_value(d.get(tid))
                if lp is None:
                    continue
                logp += lp
                n += 1

            avg_nll = -logp / max(n, 1)
            ppl = math.exp(avg_nll)
            results.append({
                "logp_answer_given_problem": logp,
                "answer_tokens_counted": n,
                "avg_nll": avg_nll,
                "ppl": ppl,
            })
        return results


class TransformersLogProbRunner:
    def __init__(
        self,
        model: str,
        *,
        device: str | None = None,
        torch_dtype: str | None = "bfloat16",
        trust_remote_code: bool = True,
        model_kwargs: dict | None = None,
    ):
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
        )
        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) and hasattr(torch, torch_dtype) else None
        kwargs = dict(model_kwargs or {})
        if device is None:
            kwargs.setdefault("device_map", "auto")
        else:
            kwargs.setdefault("device_map", {"": device})
        if dtype is not None:
            kwargs.setdefault("torch_dtype", dtype)
        kwargs.setdefault("trust_remote_code", trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.model.eval()

    def _apply_chat_template(self, messages: List[Dict[str, str]], *, tokenize: bool) -> List[int]:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise AttributeError("Tokenizer does not support chat template")

        sig = inspect.signature(self.tokenizer.apply_chat_template)
        kwargs: dict[str, Any] = {"tokenize": tokenize}
        if "add_generation_prompt" in sig.parameters:
            kwargs["add_generation_prompt"] = False
        if "continue_final_message" in sig.parameters:
            kwargs["continue_final_message"] = True
        if "enable_thinking" in sig.parameters:
            kwargs["enable_thinking"] = True

        res = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(res, dict):
            return res["input_ids"]
        return res

    def _build_inputs(self, problem: str, solution: str, sep: str) -> tuple[List[int], int]:
        system_message = (
            "You are a mathematician. Solve the problem."
            "## Style preferences (keep them light; do not change your underlying approach):"
            "- Treat `current_solution`/`ref` as correct established premises and build directly on them."
            "- Start immediately with the next logical derivation. Do not restate the problem or re-summarize what has already been established."
            "- Write as continuous mathematical prose (no section headers, no “Step 1/2/3”)."
            "- Avoid repeating the same conditions. If you must reference a prior premise, do it minimally (e.g., “from the previous inequality …”)."
        )
        user_message = f"Solve the Problem:\n{problem}"
        try:
            prefix_messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
            full_messages = prefix_messages + [{"role": "assistant", "content": solution}]
            prefix_ids = self._apply_chat_template(prefix_messages, tokenize=True)
            full_ids = self._apply_chat_template(full_messages, tokenize=True)
            return full_ids, len(prefix_ids)
        except Exception:
            prefix = problem + sep
            full = prefix + solution
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
            full_ids = self.tokenizer(full, add_special_tokens=False).input_ids
            return full_ids, len(prefix_ids)

    def score(self, problem: str, solution: str, sep: str = "\n\n") -> Dict[str, Any]:
        input_ids, boundary = self._build_inputs(problem, solution, sep)
        if not input_ids:
            return {
                "logp_answer_given_problem": 0.0,
                "answer_tokens_counted": 0,
                "avg_nll": None,
                "ppl": None,
            }

        device = next(self.model.parameters()).device
        input_tensor = torch.tensor([input_ids], device=device)
        with torch.inference_mode():
            logits = self.model(input_tensor).logits
            log_probs = torch.log_softmax(logits, dim=-1)

        start = max(boundary, 1)
        logp = 0.0
        n = 0
        for i in range(start, len(input_ids)):
            tok_id = input_ids[i]
            lp = log_probs[0, i - 1, tok_id].item()
            logp += lp
            n += 1

        avg_nll = -logp / max(n, 1)
        ppl = math.exp(avg_nll) if n > 0 else None

        return {
            "logp_answer_given_problem": logp,
            "answer_tokens_counted": n,
            "avg_nll": avg_nll,
            "ppl": ppl,
        }

    def score_batch(self, problems: List[str], solutions: List[str], sep: str = "\n\n") -> List[Dict[str, Any]]:
        assert len(problems) == len(solutions)
        return [self.score(p, s, sep=sep) for p, s in zip(problems, solutions)]

Message = Dict[str, str]
PackedPrompt = Dict[str, Any]

class DEEPSEEK_API_runner:
    def __init__(self, max_workers_default: int = 16):
        self.model_name = "deepseek-reasoner"
        self.api_key = "sk-d4cf7bbc94f74f0795a309e3be8810de"
        self.base_url = "https://api.deepseek.com/beta"
        self.base_url_beta = "https://api.deepseek.com/beta"
        self.default_params = dict(Config["judge_sampling_params"])
        self.max_workers_default = max_workers_default

        # 每个线程一个 client
        self._tls = threading.local()
        self._tls_beta = threading.local()

        # deepseek-reasoner 不支持这些参数：部分“无效但不报错”，logprobs 会直接报错 :contentReference[oaicite:2]{index=2}
        for k in ["temperature", "top_p", "presence_penalty", "frequency_penalty", "logprobs", "top_logprobs"]:
            self.default_params.pop(k, None)

    def _get_client(self, max_workers: int) -> OpenAI:
        """Thread-local OpenAI client，避免多线程复用同一个 httpx.Client 出幺蛾子。"""
        if getattr(self._tls, "client", None) is None:
            limits = httpx.Limits(
                max_connections=max(32, max_workers * 4),
                max_keepalive_connections=max(16, max_workers * 2),
                keepalive_expiry=30,
            )
            http_client = httpx.Client(
                timeout=httpx.Timeout(600.0, connect=10.0),
                limits=limits,
                http2=True,
            )
            # OpenAI SDK 支持传自定义 http_client :contentReference[oaicite:4]{index=4}
            self._tls.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=http_client,
            )
        return self._tls.client

    def _get_beta_client(self, max_workers: int) -> OpenAI:
        """Thread-local OpenAI client for /beta endpoints (needed for /completions + logprobs)."""
        if getattr(self._tls_beta, "client", None) is None:
            limits = httpx.Limits(
                max_connections=max(32, max_workers * 4),
                max_keepalive_connections=max(16, max_workers * 2),
                keepalive_expiry=30,
            )
            http_client = httpx.Client(
                timeout=httpx.Timeout(600.0, connect=10.0),
                limits=limits,
                http2=True,
            )
            self._tls_beta.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url_beta,
                http_client=http_client,
            )
        return self._tls_beta.client

    def _looks_like_json_schema(self, obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        t = obj.get("type")
        if t in {"object", "array", "string", "number", "integer", "boolean", "null"}:
            return True
        if "properties" in obj or "$schema" in obj or "required" in obj:
            return True
        return False

    def _normalize_messages(self, prompt: Union[str, List[Dict[str, str]], Dict[str, Any]]) -> List[Dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, dict) and "messages" in prompt:
            return prompt["messages"]
        if isinstance(prompt, list):
            return prompt
        raise ValueError("Unsupported prompt format")

    def generate_one(
        self,
        prompt: Union[str, List[Dict[str, str]], Dict[str, Any]],
        extra_params: dict | None = None,
        *,
        max_workers_hint: int = 8,
    ) -> Dict[str, str]:
        params = dict(self.default_params)

        schema = None
        if extra_params:
            if self._looks_like_json_schema(extra_params):
                schema = extra_params
            else:
                params.update(extra_params)

        messages = self._normalize_messages(prompt)

        # JSON Output：需要 response_format + prompt 里出现 "json" + 例子，否则可能不稳定 :contentReference[oaicite:5]{index=5}
        if schema is not None:
            params.setdefault("response_format", {"type": "json_object"})

        client = self._get_client(max_workers_hint)
        print(f"[DEBUG][Thread {threading.get_ident()}] Sending request to DEEPSEEK API...")
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **params,
            extra_body={"thinking": {"type": "enabled"}},  # 你原来的
        )

        msg = resp.choices[0].message
        #print(msg)
        #print(reasoning)
        reasoning = getattr(msg, "reasoning_content", "") or ""
        content = getattr(msg, "content", "") or ""
        return {"reasoning": reasoning, "content": content}

    def generate(
        self,
        prompts: List[Union[str, List[Dict[str, str]], Dict[str, Any]]],
        extra_params: dict | None = None,
        max_workers: int | None = None,
    ) -> Tuple[List[str], List[str]]:
        if max_workers is None:
            max_workers = self.max_workers_default

        n = len(prompts)
        max_workers = max(1, min(max_workers, n))

        reasonings = [""] * n
        contents = [""] * n

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut2i = {
                executor.submit(self.generate_one, p, extra_params, max_workers_hint=max_workers): i
                for i, p in enumerate(prompts)
            }
            for fut in as_completed(fut2i):
                i = fut2i[fut]
                try:
                    r = fut.result()
                    reasonings[i] = r["reasoning"]
                    contents[i] = r["content"]
                except Exception as e:
                    reasonings[i] = ""
                    contents[i] = f"<Error: {e}>"

        return reasonings, contents

    def score(self, problem: str, answer: str, sep: str = "\n\n", logprobs_k: int = 1) -> Dict[str, Any]:
        prefix = problem + sep
        full = prefix + answer
        client = self._get_beta_client(self.max_workers_default)
        # request payload per DeepSeek /completions schema: prompt, logprobs, max_tokens
        def _call(max_tokens: int):
            return client.completions.create(
                model=self.model_name,
                prompt=full,
                logprobs=logprobs_k,   # 0~20 in docs; 1 is enough for scoring  :contentReference[oaicite:6]{index=6}
                max_tokens=max_tokens, # try 0 for pure scoring; fallback to 1 if needed
                temperature=0,         # keep deterministic if any completion happens
            )

        try:
            resp = _call(max_tokens=0)
        except Exception:
            # fallback: generate 1 token; we'll ignore anything beyond len(full)
            resp = _call(max_tokens=1)

        choice = resp.choices[0]
        lp = choice.logprobs
        if lp is None:
            return {
                "logp_answer_given_problem": None,
                "answer_tokens_counted": 0,
                "avg_nll": None,
                "ppl": None,
                "error": "No logprobs returned. Check /beta /completions support for logprobs without echo.",
            }

        tokens = lp.get("tokens", [])
        token_logprobs = lp.get("token_logprobs", [])
        offsets = lp.get("text_offset", [])

        # sum logprobs for answer tokens only (offset >= len(prefix))
        # also ignore any generated completion tokens by enforcing offset < len(full)
        logp = 0.0
        n = 0
        for tok, lprob, off in zip(tokens, token_logprobs, offsets):
            if off is None:
                continue
            if off < len(prefix):
                continue
            if off >= len(full):
                # this is outside our provided prompt (extra generated token if max_tokens=1)
                continue
            if lprob is None:
                # first token often has None logprob
                continue
            logp += float(lprob)
            n += 1

        avg_nll = -logp / max(n, 1)
        ppl = math.exp(avg_nll)

        return {
            "logp_answer_given_problem": logp,
            "answer_tokens_counted": n,
            "avg_nll": avg_nll,
            "ppl": ppl,
        }



class DOUBAO_deepseek_API_runner:
    def __init__(
        self,
        max_concurrent_tasks: int = 32,
        timeout_sec: int = 24 * 3600,
        debug: bool = True,
        debug_max_chars: int = 4000,
    ):
        self.model_name = "ep-bi-20260112204923-vml7s"
        self.default_params = dict(Config["judge_sampling_params"])
        self.max_concurrent_tasks = max_concurrent_tasks
        self.timeout_sec = timeout_sec
        self.debug = debug
        self.debug_max_chars = debug_max_chars

        for k in ["temperature", "top_p", "presence_penalty", "frequency_penalty", "logprobs", "top_logprobs"]:
            self.default_params.pop(k, None)

    def _truncate(self, s: str, max_chars: int | None = None) -> str:
        lim = self.debug_max_chars if max_chars is None else max_chars
        if lim <= 0:
            return ""
        if len(s) <= lim:
            return s
        return s[:lim] + f"\n...<truncated, total_chars={len(s)}>"

    def _safe_json(self, obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return repr(obj)

    def _validate_req(self, req: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        if not isinstance(req, dict):
            return ["req 不是 dict"]

        if not req.get("model"):
            errors.append("缺少 model")

        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            errors.append("messages 必须是非空 list")
        else:
            for idx, m in enumerate(messages):
                if not isinstance(m, dict):
                    errors.append(f"messages[{idx}] 不是 dict")
                    continue
                role = m.get("role")
                if role not in {"system", "user", "assistant", "tool"}:
                    errors.append(f"messages[{idx}].role 非法: {role!r}")
                if "content" not in m:
                    errors.append(f"messages[{idx}] 缺少 content")

        rf = req.get("response_format")
        if rf is not None:
            if not isinstance(rf, dict) or rf.get("type") not in {"json_object", "text"}:
                errors.append(f"response_format 非法: {rf!r}")

        eb = req.get("extra_body")
        if eb is not None and not isinstance(eb, dict):
            errors.append("extra_body 必须是 dict")

        return errors

    def _looks_like_json_schema(self, obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        t = obj.get("type")
        if t in {"object", "array", "string", "number", "integer", "boolean", "null"}:
            return True
        if "properties" in obj or "$schema" in obj or "required" in obj:
            return True
        return False

    def _normalize_messages(self, prompt: Union[str, List[Dict[str, str]], Dict[str, Any]]) -> List[Dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, dict) and "messages" in prompt:
            return prompt["messages"]
        if isinstance(prompt, list):
            return prompt
        raise ValueError("Unsupported prompt format")

    def _build_one_request(
        self,
        prompt: Union[str, List[Dict[str, str]], Dict[str, Any]],
        extra_params: dict | None,
    ) -> Dict[str, Any]:
        params = dict(self.default_params)

        schema = None
        if extra_params:
            if self._looks_like_json_schema(extra_params):
                schema = extra_params
            else:
                params.update(extra_params)

        if schema is not None:
            params.setdefault("response_format", {"type": "json_object"})

        messages = self._normalize_messages(prompt)

        req: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            ##**params,
            "extra_body": {"thinking": {"type": "enabled"}}, 
        }

        return req

    async def _worker(
        self,
        worker_id: int,
        client: AsyncArk,
        q: "asyncio.Queue[Tuple[int, Dict[str, Any]]]",
        reasonings: List[str],
        contents: List[str],
    ):
        
        while True:
            i, req = await q.get()
            try:
                print(f"[Worker {worker_id}] Processing task {i}")
                resp = await client.batch.chat.completions.create(**req)
                print(f"[Worker {worker_id}] Finished task {i}")
                print("End task", i, "\n", resp)
                msg = resp.choices[0].message
                reasonings[i] = getattr(msg, "reasoning_content", "") or ""
                contents[i] = getattr(msg, "content", "") or ""
            except Exception as e:
                reasonings[i] = ""
                contents[i] = f"<Error: {e}>"
                print(e, file=sys.stderr)
            finally:
                q.task_done()

    async def generate_async(
        self,
        prompts: List[Union[str, List[Dict[str, str]], Dict[str, Any]]],
        extra_params: dict | None = None,
        max_concurrent_tasks: int | None = None,
    ) -> Tuple[List[str], List[str]]:
        start = time.time()
        n = len(prompts)
        if n == 0:
            return [], []

        k = max_concurrent_tasks or self.max_concurrent_tasks
        k = max(1, min(k, n))

        reasonings = [""] * n
        contents = [""] * n

        client = AsyncArk(
            api_key="fcd5e288-6d51-4d03-a177-2cd591f80bf9",
            timeout=self.timeout_sec,
        )

        q: "asyncio.Queue[Tuple[int, Dict[str, Any]]]" = asyncio.Queue()

        for i, p in enumerate(prompts):
            req = self._build_one_request(p, extra_params)
            await q.put((i, req))

        workers = [asyncio.create_task(self._worker(wid, client, q, reasonings, contents)) for wid in range(k)]

        await q.join()

        for t in workers:
            t.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        await client.close()
        end = time.time()
        print(f"Total time: {end - start}, Total task: {n}")
        return reasonings, contents

    def generate(
        self,
        prompts: List[Union[str, List[Dict[str, str]], Dict[str, Any]]],
        extra_params: dict | None = None,
        max_workers: int | None = None, 
    ) -> Tuple[List[str], List[str]]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError("检测到正在运行的 event loop：请改用 `await runner.generate_async(...)`")
        return asyncio.run(self.generate_async(prompts, extra_params=extra_params, max_concurrent_tasks=max_workers))


class Kimi_API_runner:
    def __init__(self):
        self.model_name = "kimi-k2-0905-preview"
        self.api_key = "sk-1hKIoFl6D5FjFC8TLZaJ3JVZ7YoY3WZCFdnYzKZdfbAkEzgb"
        self.base_url = "https://api.moonshot.cn/v1"
        self._tls = threading.local()
        
    def _get_client(self, max_workers: int) -> OpenAI:
        """Thread-local OpenAI client，避免多线程复用同一个 httpx.Client 出幺蛾子。"""
        if getattr(self._tls, "client", None) is None:
            limits = httpx.Limits(
                max_connections=max(32, max_workers * 4),
                max_keepalive_connections=max(16, max_workers * 2),
                keepalive_expiry=30,
            )
            http_client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(600.0, connect=10.0),
                limits=limits,
                http2=True,
            )
            self._tls.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=http_client,
            )
        return self._tls.client
    
    def generate_one(
        self,
        prompt: Union[str, List[Dict[str, str]], Dict[str, Any]],
        max_workers_hint: int = 8,
    ) -> Dict[str, str]:
        messages = prompt

        client = self._get_client(max_workers_hint)
        print(f"[DEBUG][Thread {threading.get_ident()}] Sending request to KIMI API...")
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024 * 8,
            temperature=0.6,
        )
        print(f"[DEBUG][Thread {threading.get_ident()}] Response: {resp}")
        msg = resp.choices[0].message
        #reasoning = getattr(msg, "reasoning_content", "") or ""
        content = getattr(msg, "content", "") or ""
        #return {"reasoning": reasoning, "content": content}
        return {"reasoning": "", "content": content}

    def generate(
        self,
        prompts: List[Union[str, List[Dict[str, str]], Dict[str, Any]]],
        max_workers: int = 16,
    ) -> Tuple[List[str], List[str]]:
        n = len(prompts)
        max_workers = max(1, min(max_workers, n))

        reasonings = [""] * n
        contents = [""] * n

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fut2i = {
                executor.submit(self.generate_one, p, max_workers_hint=max_workers): i
                for i, p in enumerate(prompts)
            }
            for fut in as_completed(fut2i):
                i = fut2i[fut]
                try:
                    r = fut.result()
                    reasonings[i] = r["reasoning"]
                    contents[i] = r["content"]
                except Exception as e:
                    reasonings[i] = ""
                    contents[i] = f"<Error: {e}>"

        return reasonings, contents
