import pytest

from videoagent.llm.errors import (
    AllProvidersFailed, AuthError, BadRequestError, ModelNotFoundError, RateLimitError,
    TransientError,
)
from videoagent.llm.fallback import FallbackChain, Provider
from videoagent.llm.providers import RetryConfig

from .llm_helpers import MSG, FakeBackend, cfg, make_chain, ok


def test_first_provider_is_used_when_healthy():
    a, b = FakeBackend(ok("from a")), FakeBackend(ok("from b"))
    chain, _ = make_chain(("a", a), ("b", b))
    assert chain.chat(MSG).text == "from a"
    assert len(a.calls) == 1 and len(b.calls) == 0


def test_rate_limit_falls_through_and_cooldown_is_respected():
    a, b = FakeBackend(RateLimitError(), ok("a back")), FakeBackend(ok("from b"))
    chain, clock = make_chain(("a", a), ("b", b))
    assert chain.chat(MSG).text == "from b"
    assert chain.status()["a"] == "cooling down (60s)"
    # still cooling: a is not even tried
    assert chain.chat(MSG).text == "from b" and len(a.calls) == 1
    clock.now += 61
    assert chain.chat(MSG).text == "a back" and len(a.calls) == 2  # recovered


def test_retry_after_header_sets_cooldown():
    a, b = FakeBackend(RateLimitError(retry_after=7)), FakeBackend(ok())
    chain, _ = make_chain(("a", a), ("b", b))
    chain.chat(MSG)
    assert chain.status()["a"] == "cooling down (7s)"


def test_repeated_429_escalates_cooldown_and_success_resets_it():
    a = FakeBackend(RateLimitError(), RateLimitError(), RateLimitError(), ok("a"))
    b = FakeBackend(ok("b"))
    chain, clock = make_chain(("a", a), ("b", b))
    for expected in ("60s", "120s", "240s"):
        chain.chat(MSG)
        assert chain.status()["a"] == f"cooling down ({expected})"
        clock.now += 1000
    assert chain.chat(MSG).text == "a"
    assert chain.providers[0].rate_limit_streak == 0


def test_auth_error_disables_provider_for_good():
    a, b = FakeBackend(AuthError("bad key")), FakeBackend(ok("b"))
    chain, clock = make_chain(("a", a), ("b", b))
    chain.chat(MSG)
    clock.now += 10_000
    chain.chat(MSG)
    assert len(a.calls) == 1
    assert "check the key" in chain.status()["a"]


def test_model_not_found_disables_and_points_to_llm_check():
    a, b = FakeBackend(ModelNotFoundError("gone")), FakeBackend(ok("b"))
    chain, _ = make_chain(("a", a), ("b", b))
    chain.chat(MSG)
    assert "llm_check" in chain.status()["a"] and "a-model" in chain.status()["a"]


def test_transient_error_is_retried_with_backoff():
    a = FakeBackend(TransientError("timeout"), ok("second try"))
    chain, clock = make_chain(("a", a))
    assert chain.chat(MSG).text == "second try"
    assert clock.sleeps == [2.0]
    assert (chain.usage.calls, chain.usage.failures) == (1, 1)


def test_exhausted_transient_moves_to_next_provider_and_sets_short_cooldown():
    a, b = FakeBackend(TransientError("5xx")), FakeBackend(ok("b"))
    chain, clock = make_chain(("a", a), ("b", b))
    assert chain.chat(MSG).text == "b"
    assert len(a.calls) == 2 and clock.sleeps == [2.0]
    assert chain.status()["a"].startswith("cooling down")


def test_json_mode_is_dropped_when_provider_rejects_it():
    a = FakeBackend(BadRequestError("response_format unsupported"), ok("{}"))
    chain, _ = make_chain(("a", a))
    chain.chat(MSG, want_json=True)
    assert a.calls[0]["response_format"] == {"type": "json_object"}
    assert a.calls[1]["response_format"] is None
    assert chain.providers[0].json_mode == "none"
    chain.chat(MSG, want_json=True)  # remembered: no JSON mode on later calls
    assert a.calls[2]["response_format"] is None
    assert len(chain.usage.records) == 2 and chain.usage.failures == 0  # not counted as a failed attempt


def test_bad_request_without_json_skips_for_this_call_only():
    a, b = FakeBackend(BadRequestError("prompt too long"), ok("a")), FakeBackend(ok("b"))
    chain, _ = make_chain(("a", a), ("b", b))
    assert chain.chat(MSG).text == "b"
    assert chain.providers[0].disabled is None
    assert chain.chat(MSG).text == "a"


def test_json_schema_mode_sends_the_schema():
    a = FakeBackend(ok("{}"))
    chain, _ = make_chain(("a", a, {"json_mode": "json_schema"}))
    chain.chat(MSG, want_json=True, json_schema={"type": "object"})
    fmt = a.calls[0]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["schema"] == {"type": "object"}


def test_json_none_mode_sends_nothing():
    a = FakeBackend(ok("{}"))
    chain, _ = make_chain(("a", a, {"json_mode": "none"}))
    chain.chat(MSG, want_json=True)
    assert a.calls[0]["response_format"] is None


def test_all_failed_reports_every_reason():
    chain, _ = make_chain(
        ("a", FakeBackend(AuthError("nope"))),
        ("b", FakeBackend(BadRequestError("too long"))),
    )
    with pytest.raises(AllProvidersFailed) as exc:
        chain.chat(MSG)
    assert set(exc.value.failures) == {"a", "b"}
    assert "check the key" in exc.value.failures["a"] and "rejected" in exc.value.failures["b"]


def test_waits_for_a_short_cooldown_when_nothing_else_is_available():
    a = FakeBackend(RateLimitError(retry_after=5), ok("after wait"))
    chain, clock = make_chain(("a", a))
    assert chain.chat(MSG).text == "after wait"
    assert clock.sleeps == [5]


def test_raises_with_retry_hint_when_the_wait_is_too_long():
    a = FakeBackend(RateLimitError(retry_after=500))
    chain, clock = make_chain(("a", a))
    with pytest.raises(AllProvidersFailed) as exc:
        chain.chat(MSG)
    assert exc.value.retry_after == pytest.approx(500)
    assert clock.sleeps == []  # did not sleep for 500 s
    with pytest.raises(AllProvidersFailed):  # still cooling: fails fast without calling again
        chain.chat(MSG)
    assert len(a.calls) == 1


def test_permanent_outage_terminates():
    """A provider that always times out must end in AllProvidersFailed, never an endless loop."""
    a = FakeBackend(TransientError("down"))
    chain, clock = make_chain(("a", a))
    with pytest.raises(AllProvidersFailed):
        chain.chat(MSG)
    assert len(a.calls) <= 3 * 2  # at most MAX_PASSES x attempts_per_provider


def test_disabled_providers_appear_in_failures_up_front():
    chain = FallbackChain(
        [Provider(cfg("a"), None, disabled="no key"), Provider(cfg("b"), FakeBackend(AuthError("x")))],
        RetryConfig(),
    )
    with pytest.raises(AllProvidersFailed) as exc:
        chain.chat(MSG)
    assert exc.value.failures["a"] == "no key"


def test_no_providers_at_all():
    with pytest.raises(AllProvidersFailed, match="no provider configured"):
        FallbackChain([]).chat(MSG)


def test_usage_records_tokens_cost_and_tag():
    a = FakeBackend(ok(prompt=2_000_000, completion=1_000_000))
    chain, _ = make_chain(("a", a, {"price_in_per_mtok": 0.1, "price_out_per_mtok": 0.4}))
    chain.chat(MSG, tag="planner")
    rec = chain.usage.records[0]
    assert (rec.provider, rec.tag, rec.ok) == ("a", "planner", True)
    assert rec.cost_usd == pytest.approx(0.6)


def test_request_options_are_passed_through():
    a = FakeBackend(ok())
    chain, _ = make_chain(("a", a))
    chain.chat(MSG, temperature=0.9, max_tokens=123)
    assert a.calls[0]["temperature"] == 0.9 and a.calls[0]["max_tokens"] == 123


def test_status_lists_all_states():
    chain = FallbackChain(
        [Provider(cfg("a"), FakeBackend(ok())), Provider(cfg("b"), None, disabled="no key")],
        RetryConfig(),
    )
    assert chain.status() == {"a": "ready", "b": "no key"}
