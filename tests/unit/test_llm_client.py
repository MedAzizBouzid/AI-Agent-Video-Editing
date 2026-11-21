import json

import pytest
from pydantic import BaseModel

from videoagent.llm import StructuredOutputError, build_client
from videoagent.llm.client import LLMClient
from videoagent.schemas import EDL

from .llm_helpers import FakeBackend, make_chain, ok


class Plan(BaseModel):
    title: str
    scenes: int


def client_with(*replies, **kw):
    backend = FakeBackend(*replies)
    chain, _ = make_chain(("a", backend))
    return LLMClient(chain, **kw), backend


def test_complete_json_happy_path():
    client, backend = client_with(ok('{"title": "Demo", "scenes": 3}'))
    plan = client.complete_json("make a plan", Plan, tag="planner")
    assert plan == Plan(title="Demo", scenes=3)
    sent = backend.calls[0]
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["messages"][0]["role"] == "system" and "JSON Schema" in sent["messages"][0]["content"]
    assert sent["messages"][-1] == {"role": "user", "content": "make a plan"}


def test_complete_json_accepts_fenced_and_chatty_replies():
    client, _ = client_with(ok('Of course!\n```json\n{"title": "T", "scenes": 1}\n```'))
    assert client.complete_json("x", Plan).scenes == 1


def test_repair_loop_feeds_back_the_exact_errors():
    client, backend = client_with(
        ok('{"title": "T", "scenes": "many"}'),
        ok('{"title": "T", "scenes": 4}'),
    )
    plan = client.complete_json("x", Plan, tag="planner")
    assert plan.scenes == 4
    repair_prompt = backend.calls[1]["messages"]
    assert repair_prompt[-2]["role"] == "assistant" and "many" in repair_prompt[-2]["content"]
    assert repair_prompt[-1]["role"] == "user"
    assert "scenes" in repair_prompt[-1]["content"] and "corrected JSON" in repair_prompt[-1]["content"]
    assert [r.tag for r in client.usage.records] == ["planner", "planner:repair"]


def test_repair_handles_non_json_reply():
    client, backend = client_with(ok("I cannot do that"), ok('{"title": "T", "scenes": 1}'))
    assert client.complete_json("x", Plan).title == "T"
    assert "no valid JSON" in backend.calls[1]["messages"][-1]["content"]


def test_gives_up_after_max_repairs_and_keeps_last_text():
    client, backend = client_with(ok('{"title": 1}'), max_repairs=2)
    with pytest.raises(StructuredOutputError) as exc:
        client.complete_json("x", Plan)
    assert len(backend.calls) == 3  # first try + 2 repairs
    assert exc.value.last_text == '{"title": 1}' and "Plan" in str(exc.value)


def test_zero_repairs_means_one_attempt():
    client, backend = client_with(ok("nope"))
    with pytest.raises(StructuredOutputError):
        client.complete_json("x", Plan, max_repairs=0)
    assert len(backend.calls) == 1


def test_truncated_reply_asks_for_a_shorter_one():
    client, backend = client_with(
        ok('{"title": "T", "scen', finish="length"), ok('{"title": "T", "scenes": 2}')
    )
    client.complete_json("x", Plan)
    assert "cut off" in backend.calls[1]["messages"][-1]["content"]


def test_system_prompt_is_merged_not_duplicated():
    client, backend = client_with(ok('{"title": "T", "scenes": 1}'), ok('{"title": "T", "scenes": 1}'))
    client.complete_json("x", Plan, system="You are a planner.")
    msgs = backend.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"].startswith("You are a planner.") and "JSON Schema" in msgs[0]["content"]

    client.complete_json(
        [{"role": "system", "content": "Existing."}, {"role": "user", "content": "x"}], Plan
    )
    msgs = backend.calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[0]["content"].startswith("Existing.")


def test_complete_free_text():
    client, backend = client_with(ok("A catchy title"))
    res = client.complete("title please", system="Be brief.", tag="copy", max_tokens=20)
    assert res.text == "A catchy title"
    assert backend.calls[0]["response_format"] is None and backend.calls[0]["max_tokens"] == 20
    assert backend.calls[0]["messages"][0] == {"role": "system", "content": "Be brief."}
    assert client.usage.records[0].tag == "copy"


def test_real_edl_schema_round_trip():
    """The actual contract the planner will use: the model returns an EDL, we get a validated EDL."""
    edl_json = json.dumps({
        "platform": "shorts", "title": "Demo",
        "clips": [{"source": "raw.mp4", "start": 1.0, "end": 5.0}],
        "captions": [{"start": 0.0, "end": 2.0, "text": "مرحبا"}],
    })
    client, backend = client_with(ok("```json\n" + edl_json + "\n```"))
    edl = client.complete_json("plan", EDL, system="You are an editor.", tag="planner")
    assert isinstance(edl, EDL) and edl.clips[0].source == "raw.mp4" and edl.captions[0].text == "مرحبا"
    assert "clips" in backend.calls[0]["messages"][0]["content"]  # schema was shown to the model


def test_schema_error_on_edl_is_actionable():
    bad = json.dumps({"platform": "myspace", "title": "T", "clips": []})
    client, backend = client_with(ok(bad), ok(json.dumps(
        {"platform": "youtube", "title": "T", "clips": [{"source": "a.mp4", "start": 0, "end": 2}]})))
    assert client.complete_json("plan", EDL).platform == "youtube"
    feedback = backend.calls[1]["messages"][-1]["content"]
    assert "platform" in feedback and "clips" in feedback


# ---- build_client ----------------------------------------------------------

def _factory(created):
    def make(cfg, key):
        created[cfg.name] = key
        return FakeBackend(ok("hello"))
    return make


def write_config(tmp_path, body):
    f = tmp_path / "p.yaml"
    f.write_text(body)
    return f


YAML = """
providers:
  - {name: free, base_url: http://a, api_key_env: KEY_A, model: m1}
  - {name: trainer, base_url: http://b, api_key_env: KEY_B, model: m2, trains_on_data: true}
  - {name: nokey, base_url: http://c, api_key_env: KEY_C, model: m3}
retry: {max_attempts_per_provider: 1}
"""


def test_build_client_wires_keys_and_reports_unavailable(tmp_path):
    created: dict = {}
    client = build_client(write_config(tmp_path, YAML), env={"KEY_A": "ka", "KEY_B": "kb"},
                          backend_factory=_factory(created), load_dotenv_file=False)
    assert created == {"free": "ka", "trainer": "kb"}
    status = client.chain.status()
    assert status["free"] == "ready" and status["trainer"] == "ready"
    assert "KEY_C is not set" in status["nokey"]
    assert client.chain.retry.max_attempts_per_provider == 1


def test_privacy_strict_excludes_training_providers(tmp_path):
    created: dict = {}
    client = build_client(write_config(tmp_path, YAML), env={"KEY_A": "ka", "KEY_B": "kb"},
                          strict_privacy=True, backend_factory=_factory(created), load_dotenv_file=False)
    assert "trainer" not in created
    assert "PRIVACY_STRICT" in client.chain.status()["trainer"]


def test_privacy_strict_read_from_environment(tmp_path):
    created: dict = {}
    build_client(write_config(tmp_path, YAML), env={"KEY_A": "ka", "KEY_B": "kb", "PRIVACY_STRICT": "1"},
                 backend_factory=_factory(created), load_dotenv_file=False)
    assert list(created) == ["free"]


def test_built_client_works_end_to_end(tmp_path):
    client = build_client(write_config(tmp_path, YAML), env={"KEY_A": "ka"},
                          backend_factory=_factory({}), load_dotenv_file=False)
    assert client.complete("hi").text == "hello"
