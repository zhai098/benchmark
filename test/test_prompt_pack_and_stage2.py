from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.prompts import pack_prompt
from tools.prompts import stage2_judge_from_cache_prompts as stage2


def test_pack_prompt_exclude_last_positions_controls_scored_indices():
    record = {
        "gen_output": ["one. two.", "three. four.", "five. six.", "seven. eight."],
        "gen_prefix": ["one", "three", "five", "seven"],
    }

    assert pack_prompt._iter_scored_prefixes(record, exclude_last_positions=2) == [
        (0, "one"),
        (1, "three"),
    ]
    assert pack_prompt._iter_scored_prefixes(record, exclude_last_positions=0) == [
        (0, "one"),
        (1, "three"),
        (2, "five"),
        (3, "seven"),
    ]


def test_stage2_requested_indices_are_unique_sorted_and_nonnegative():
    requests = [
        {"idx": 2},
        {"idx": "0"},
        {"idx": 2},
        {"idx": -1},
        {"idx": "not-an-index"},
        {},
    ]

    assert stage2._requested_indices(requests) == [0, 2]
