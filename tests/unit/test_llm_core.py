import pytest
from pydantic import BaseModel, ValidationError

from videoagent.llm.errors import AllProvidersFailed
from videoagent.llm.providers import ProviderConfig, load_llm_config, unavailable_reason
from videoagent.llm.structured import (
    extract_json, format_validation_error, schema_instructions,
)
from videoagent.llm.usage import UsageLog, UsageRecord, cost_usd, estimate_tokens


# ---- extract_json ----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('Sure! Here you go:\n```json\n{"a": [1, 2]}\n```\nHope it helps', {"a": [1, 2]}),
    ('```\n{"a": 1}\n```', {"a": 1}),
    ('<think>I should output {"wrong": true}</think>\n{"right": true}', {"right": True}),
    ('Result: {"a": {"b": "has } brace"}} trailing text', {"a": {"b": "has } brace"}}),
    ('[1, 2, 3]', [1, 2, 3]),
    ('{"ar": "مرحبا"}', {"ar": "مرحبا"}),
    ('{broken} then {"ok": 1}', {"ok": 1}),
])
def test_extract_json_tolerates_model_noise(text, expected):
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", ["no json here", "", '{"unterminated": ', "just { braces"])
def test_extract_json_raises_on_garbage(text):
    with pytest.raises(ValueError):
        extract_json(text)


class Item(BaseModel):
    name: str
    count: int


def test_schema_instructions_mentions_fields_and_forbids_prose():
    text = schema_instructions(Item)
    assert "name" in text and "count" in text and "JSON" in text and "no markdown" in text


def test_format_validation_error_is_readable_and_capped():
    with pytest.raises(ValidationError) as exc:
        Item(name=3, count="x")  # type: ignore[arg-type]
    text = format_validation_error(exc.value)
    assert "- name:" in text and "- count:" in text

    class Many(BaseModel):
        items: list[Item]

    with pytest.raises(ValidationError) as exc2:
        Many(items=[{"name": 1, "count": "a"} for _ in range(10)])
    assert "more" in format_validation_error(exc2.value, limit=3)
    assert format_validation_error(ValueError("boom")) == "- boom"


# ---- providers config ------------------------------------------------------

def test_shipped_config_loads():
    config = load_llm_config()
    assert [p.name for p in config.providers] == ["openrouter", "gemini"]
    assert config.retry.max_attempts_per_provider == 2
    assert next(p for p in config.providers if p.name == "gemini").trains_on_data is True


def test_config_rejects_unknown_keys(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("providers:\n  - {name: a, base_url: u, api_key_env: K, model: m, trian_on_data: true}\n")
    with pytest.raises(ValidationError):
        load_llm_config(f)


def test_unavailable_reasons():
    ok = ProviderConfig(name="a", base_url="u", api_key_env="K", model="m")
    assert unavailable_reason(ok, {"K": "secret"}) is None
    assert "K is not set" in unavailable_reason(ok, {})
    assert "K is not set" in unavailable_reason(ok, {"K": ""})
    placeholder = ok.model_copy(update={"model": "REPLACE_WITH_ID"})
    assert "placeholder" in unavailable_reason(placeholder, {"K": "x"})
    trains = ok.model_copy(update={"trains_on_data": True})
    assert unavailable_reason(trains, {"K": "x"}) is None
    assert "PRIVACY_STRICT" in unavailable_reason(trains, {"K": "x"}, strict_privacy=True)


# ---- usage -----------------------------------------------------------------

def test_cost_and_token_helpers():
    assert cost_usd(1_000_000, 500_000, 0.10, 0.40) == pytest.approx(0.30)
    assert cost_usd(1000, 1000, 0, 0) == 0
    assert estimate_tokens("a" * 400) == 100 and estimate_tokens("") == 1


def test_usage_log_totals_and_grouping():
    log = UsageLog()
    log.add(UsageRecord("a", "m", "planner", True, 100, 50, cost_usd=0.01))
    log.add(UsageRecord("a", "m", "planner:repair", True, 120, 60, cost_usd=0.02))
    log.add(UsageRecord("b", "m2", "critic", True, 10, 5))
    log.add(UsageRecord("a", "m", "planner", False, error="429"))
    assert (log.calls, log.failures) == (3, 1)
    assert (log.prompt_tokens, log.completion_tokens) == (230, 115)
    assert log.cost_usd == pytest.approx(0.03)
    by_provider = log.by("provider")
    assert by_provider["a"]["calls"] == 2 and by_provider["a"]["failures"] == 1
    assert log.by("tag")["planner:repair"]["tokens"] == 180
    assert "3 calls, 1 failed" in log.summary()


def test_all_providers_failed_message():
    err = AllProvidersFailed({"a": "rate limited", "b": "no key"}, retry_after=42)
    assert "a: rate limited" in str(err) and "42s" in str(err)
    assert "no provider configured" in str(AllProvidersFailed({}))
