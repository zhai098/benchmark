from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
import time
import copy 
from typing import List, Union, Dict, Any, Tuple
from openai import OpenAI
from config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, threading
import httpx
import sys
import asyncio
import json
from typing import Any, Dict, List, Tuple, Union
from volcenginesdkarkruntime import AsyncArk

class ApiRunner:
    def __init__(self, max_concurrent_tasks: int = 10):
        self.model_name = "ep-bi-20260112204923-vml7s"
        self.timeout_sec = 300
        self.max_concurrent_tasks = max_concurrent_tasks
        self.i = 0

    def _build_one_request(
        self,
    ) -> Dict[str, Any]:
        # params = dict(self.default_params)

        # schema = None
        # if extra_params:
        #     if self._looks_like_json_schema(extra_params):
        #         schema = extra_params
        #     else:
        #         params.update(extra_params)

        # if schema is not None:
        #     params.setdefault("response_format", {"type": "json_object"})

        # messages = self._normalize_messages(prompt)

        req: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": "你好，请你重复'{i}'这个数字{i}遍".format(i=self.i)},
            ],
            "extra_body": {"thinking": {"type": "disabled"}}, 
        }
        self.i += 1
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
                print("Begin task", i)
                resp = await client.batch.chat.completions.create(**req)
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
        max_concurrent_tasks
    ) -> Tuple[List[str], List[str]]:
        start = time.time()
        n = 100
        if n == 0:
            return [], []

        k = max_concurrent_tasks or self.max_concurrent_tasks
        k = max(1, min(k, n))

        reasonings = [""] * n
        contents = [""] * n

        client = AsyncArk(
            api_key="fcd5e288-6d51-4d03-a177-2cd591f80bf9", #os.environ.get("ARK_API_KEY"),
            timeout=self.timeout_sec,
        )

        q = asyncio.Queue()

        for i, p in enumerate(range(100)):
            req = self._build_one_request()
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
        max_workers, 
    ) -> Tuple[List[str], List[str]]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError("检测到正在运行的 event loop：请改用 `await runner.generate_async(...)`")
        return asyncio.run(self.generate_async(max_concurrent_tasks=max_workers))

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
            "messages": [
                {"role": "user", "content": "你好，请你重复'1'这个数字10遍"},
            ],
            **params,
            "extra_body": {"thinking": {"type": "disabled"}}, 
        }

        if self.debug:
            print("\n[DOUBAO][DEBUG] input prompt:")
            print(self._truncate(self._safe_json(prompt)))
            print("\n[DOUBAO][DEBUG] built req:")
            print(self._truncate(self._safe_json(req)))
            errs = self._validate_req(req)
            if errs:
                print("\n[DOUBAO][DEBUG] req validation errors:")
                for e in errs:
                    print("-", e)
                raise ValueError("DOUBAO req 不合法: " + "; ".join(errs))

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
        n = 100

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
        prompts: List[Union[str, List[Dict[str, str]], Dict[str, Any]]] | None = None,
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




if __name__ == "__main__":
    runner = ApiRunner()
    runner1 = DOUBAO_deepseek_API_runner()
    runner.generate(max_workers=10)