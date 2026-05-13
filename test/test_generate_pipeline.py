import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from generate import generate_case


WORKFLOW_PURIFIED = Path(
    "generate_pipeline_test_data/purified_cases.jsonl"
)


def _workflow_case(case_id: str) -> dict:
    for line in WORKFLOW_PURIFIED.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("case_id") == case_id:
            return row
    raise KeyError(case_id)


class FakeReasoningModel:
    def __init__(self) -> None:
        self.model_name = "fake-reasoner"
        self.last_prompts = None
        self.last_schema = None

    def generate(self, prompts, schema):
        self.last_prompts = prompts
        self.last_schema = schema
        return [f"Generated prefix {idx}. More detail follows." for idx, _ in enumerate(prompts, start=1)]



def test_generate_case_prefers_annotation_reference_steps_from_workflow_outputs(monkeypatch):
    record = _workflow_case("q-21")
    model = FakeReasoningModel()

    monkeypatch.setattr(
        "generate.processor.sentence_split_en",
        lambda text: [part.strip() for part in text.split(".") if part.strip()],
    )

    result = generate_case(record, model)

    assert model.last_schema is None
    assert len(model.last_prompts) == len(record["reference_steps"])
    assert result["ref_steps"][0] == record["reference_steps"][0]["text"]
    assert result["ref_steps"][-1] == record["reference_steps"][-1]["text"]
    assert result["ref_steps"][0] != record["segments"][0]["content"]

    first_prompt = model.last_prompts[0]
    assert first_prompt[0]["role"] == "system"
    assert first_prompt[1]["role"] == "user"
    assert first_prompt[2]["role"] == "assistant"
    assert first_prompt[2]["prefix"] is True
    assert "partial" not in first_prompt[2]
    assert first_prompt[2]["content"] == record["reference_steps"][0]["text"]

    assert result["gen_output"][0].startswith("Generated prefix 1")
    assert result["gen_prefix"][0].startswith("Generated prefix 1")


def test_generate_case_does_not_fallback_to_benchmark_segments_without_annotation_steps(monkeypatch):
    record = _workflow_case("q-1")
    model = FakeReasoningModel()

    monkeypatch.setattr(
        "generate.processor.sentence_split_en",
        lambda text: [part.strip() for part in text.split(".") if part.strip()],
    )

    result = generate_case(record, model)

    assert record["reference_steps"] == []
    assert record["segments"]
    assert model.last_schema is None
    assert model.last_prompts == []
    assert result["ref_steps"] == []
    assert result["gen_output"] == []
    assert result["gen_prefix"] == []


def test_generate_output_fields_can_forward_reference_metadata_from_workflow_inputs():
    record = _workflow_case("q-21")

    forwarded = {
        "steps": record.get("reference_steps") or record.get("steps", []),
        "claims_by_step": record.get("reference_claims_by_step") or record.get("claims_by_step", []),
        "step_dependencies": record.get("reference_step_dependencies") or record.get("step_dependencies", {}),
    }

    assert forwarded["steps"] == record["reference_steps"]
    assert forwarded["claims_by_step"] == record["reference_claims_by_step"]
    assert forwarded["step_dependencies"] == record["reference_step_dependencies"]


def test_vllm_runner_source_contains_message_to_text_prompt_normalization():
    source = Path("runner.py").read_text(encoding="utf-8")

    assert "def _chat_messages_to_prompt_text" in source
    assert "def _normalize_generate_prompt" in source
    assert 'if isinstance(prompt, dict) and "messages" in prompt' in source
    assert 'if self._is_chat_message(first):' in source
    assert 'if isinstance(first, list) and first and self._is_chat_message(first[0]):' in source
    assert 'kwargs["add_generation_prompt"] = not continue_final_message' in source
    assert 'kwargs["continue_final_message"] = True' in source


def test_generate_prompt_pack_auto_is_vllm_messages_without_api_partial():
    from tools.prompts.build_generate_prompt_pack import _format_prompt

    prompt, actual_format = _format_prompt("Find x.", "We have x=1.", "kimi-k2.5", "auto")

    assert actual_format == "vllm-messages"
    assert prompt[-1]["role"] == "assistant"
    assert prompt[-1]["prefix"] is True
    assert "partial" not in prompt[-1]


def test_prompt_pack_runner_is_vllm_only():
    source = Path("tools/prompts/run_generate_prompt_pack.py").read_text(encoding="utf-8")

    assert 'choices=["vllm"]' in source
    assert "Kimi_API_runner" not in source
    assert "DEEPSEEK_API_runner" not in source


def test_generate_py_builds_vllm_runner_only():
    source = Path("generate.py").read_text(encoding="utf-8")

    assert "from runner import VLLMRunner" in source
    assert "DEEPSEEK_API_runner" not in source
    assert "return reasoning_model.generate(list(prompts), None)" in source
