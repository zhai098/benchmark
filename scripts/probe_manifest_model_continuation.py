#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe whether a model tokenizer can render assistant-prefix continuation.

This is intentionally tokenizer-only. It checks the exact chat-template path
used by completed-annotation generation before any GPU model loading starts.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner import _load_generation_tokenizer  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def signature_accepts_kwargs(callable_obj: Any) -> bool:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


def filter_kwargs(tokenizer: Any, kwargs: Dict[str, Any], *, base: bool = False) -> Dict[str, Any]:
    sig = inspect.signature(tokenizer.apply_chat_template)
    accepts_kwargs = signature_accepts_kwargs(tokenizer.apply_chat_template)
    filtered: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in sig.parameters:
            filtered[key] = value
        elif accepts_kwargs and not (base and key in {"add_generation_prompt"}):
            filtered[key] = value
    return filtered


def normalize_messages(messages: List[Dict[str, str]], *, no_system_role: bool) -> List[Dict[str, str]]:
    normalized = [dict(message) for message in messages]
    if (
        no_system_role
        and len(normalized) >= 2
        and normalized[0].get("role") == "system"
        and normalized[1].get("role") == "user"
    ):
        system_text = str(normalized[0].get("content") or "")
        user_text = str(normalized[1].get("content") or "")
        merged_user = dict(normalized[1])
        merged_user["content"] = f"{system_text}\n\n{user_text}".strip()
        normalized = [merged_user] + normalized[2:]
    return normalized


def classify_failure(error_text: str, supports_continue: bool) -> Dict[str, str]:
    lower = error_text.lower()
    if not supports_continue or "continue_final_message" in lower and "unexpected" in lower:
        return {
            "issue_type": "tokenizer_does_not_support_continue_final_message",
            "likely_cause": "tokenizer_or_chat_template_capability",
        }
    if "jinja" in lower or "template" in lower or "chat_template" in lower:
        return {
            "issue_type": "continuation_render_error",
            "likely_cause": "tokenizer_chat_template_runtime",
        }
    return {
        "issue_type": "continuation_probe_error",
        "likely_cause": "code_or_tokenizer_runtime",
    }


def probe(model_path: str, model_name: str, chat_template_kwargs: Dict[str, Any], no_system_role: bool) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "model_name": model_name,
        "model_path": model_path,
        "checked_at_utc": now_iso(),
        "chat_template_kwargs_requested": chat_template_kwargs,
        "no_system_role": no_system_role,
    }
    tokenizer = _load_generation_tokenizer(model_path)
    result["tokenizer_class"] = tokenizer.__class__.__name__
    result["has_chat_template"] = bool(getattr(tokenizer, "chat_template", None))
    if not hasattr(tokenizer, "apply_chat_template"):
        result.update(
            {
                "ok": False,
                "issue_type": "tokenizer_does_not_support_chat_template",
                "likely_cause": "tokenizer_or_chat_template_capability",
            }
        )
        return result

    sig = inspect.signature(tokenizer.apply_chat_template)
    accepts_kwargs = signature_accepts_kwargs(tokenizer.apply_chat_template)
    supports_continue = "continue_final_message" in sig.parameters or accepts_kwargs
    result["apply_chat_template_signature"] = str(sig)
    result["apply_chat_template_accepts_kwargs"] = accepts_kwargs
    result["supports_continue_final_message"] = supports_continue
    result["chat_template_excerpt"] = (getattr(tokenizer, "chat_template", "") or "")[:800]
    if not supports_continue:
        result.update(
            {
                "ok": False,
                "issue_type": "tokenizer_does_not_support_continue_final_message",
                "likely_cause": "tokenizer_or_chat_template_capability",
            }
        )
        return result

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a mathematician. Continue the proof."},
        {"role": "user", "content": "Solve the Problem:\nWhat is 1+1?"},
        {"role": "assistant", "content": "We have 1+1="},
    ]
    messages = normalize_messages(messages, no_system_role=no_system_role)
    base_kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
        "continue_final_message": True,
    }
    kwargs = dict(filter_kwargs(tokenizer, chat_template_kwargs))
    kwargs.update(filter_kwargs(tokenizer, base_kwargs, base=True))
    result["chat_template_kwargs_used"] = kwargs

    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
        rendered_text = rendered if isinstance(rendered, str) else str(rendered)
        result.update(
            {
                "ok": bool(rendered_text.strip()),
                "rendered_length": len(rendered_text),
                "rendered_tail": rendered_text[-500:],
                "assistant_prefix_present": "We have 1+1=" in rendered_text,
            }
        )
        if not result["ok"]:
            result.update(
                {
                    "issue_type": "empty_rendered_prompt",
                    "likely_cause": "tokenizer_chat_template_runtime",
                }
            )
    except Exception as exc:
        failure = classify_failure(str(exc), supports_continue)
        result.update(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                **failure,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe tokenizer assistant continuation support.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--chat-template-kwargs-json", default="{}")
    parser.add_argument("--no-system-role", action="store_true")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    try:
        chat_template_kwargs = json.loads(args.chat_template_kwargs_json or "{}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --chat-template-kwargs-json: {exc}") from exc
    if not isinstance(chat_template_kwargs, dict):
        raise SystemExit("--chat-template-kwargs-json must decode to an object")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = probe(args.model_path, args.model_name, chat_template_kwargs, args.no_system_role)
    except Exception as exc:
        result = {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "checked_at_utc": now_iso(),
            "ok": False,
            "issue_type": "continuation_probe_exception",
            "likely_cause": "code_or_tokenizer_runtime",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result.get("ok") else 2)


if __name__ == "__main__":
    main()
