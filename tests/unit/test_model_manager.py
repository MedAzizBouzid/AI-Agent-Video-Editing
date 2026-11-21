import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from videoagent.perception import PerceptionConfig, build_perception_doc
from videoagent.perception.transcribe import WHISPER_VRAM_MB, register_whisper
from videoagent.runtime import ModelBusyError, ModelError, ModelManager
from videoagent.runtime.model_manager import (
    DEFAULT_BUDGET_MB, default_budget_mb, free_gpu_memory, is_oom, parse_nvidia_smi,
)


class Rig:
    """A manager whose models are plain strings, recording loads, unloads and cleanups."""

    def __init__(self, budget=5000, sizes=None):
        self.loads, self.unloads, self.cleanups = [], [], 0
        self._tick = 0

        def clock():
            self._tick += 1
            return float(self._tick)

        def cleanup():
            self.cleanups += 1

        self.m = ModelManager(budget, cleanup=cleanup, clock=clock)
        for name, mb in (sizes or {"a": 2000, "b": 600, "c": 3000, "big": 5000}).items():
            self.add(name, mb)

    def add(self, name, mb, **kw):
        def loader(name=name):
            self.loads.append(name)
            return f"model-{name}"

        self.m.register(name, loader, vram_mb=mb, unloader=lambda model, n=name: self.unloads.append(n), **kw)

    def touch(self, *names):
        for n in names:
            with self.m.use(n):
                pass


def test_loads_lazily_and_reuses():
    r = Rig()
    assert r.loads == [] and r.m.loaded() == []
    with r.m.use("a") as model:
        assert model == "model-a"
    r.touch("a")
    assert r.loads == ["a"] and r.m.loaded() == ["a"]


def test_registering_twice_keeps_the_first():
    r = Rig()
    r.m.register("a", lambda: "different", vram_mb=1)
    with r.m.use("a") as model:
        assert model == "model-a"


def test_a_model_larger_than_the_budget_is_rejected_with_advice():
    m = ModelManager(1000)
    with pytest.raises(ValueError, match="VIDEOAGENT_VRAM_MB"):
        m.register("huge", lambda: 1, vram_mb=2000)


def test_least_recently_used_model_is_evicted_first():
    r = Rig()
    r.touch("a", "b")                     # 2600 MB loaded
    r.touch("c")                          # needs 3000: 2600 + 3000 > 5000, evict LRU = a
    assert r.m.loaded() == ["b", "c"] and r.unloads == ["a"] and r.cleanups == 1


def test_recent_use_protects_a_model_from_eviction():
    r = Rig()
    r.touch("a", "b", "a")                # a is now more recent than b
    r.touch("c")
    assert r.m.loaded() == ["a", "c"] and r.unloads == ["b"]


def test_models_that_fit_together_stay_loaded():
    r = Rig()
    r.touch("a", "b")
    assert r.m.loaded() == ["a", "b"] and r.unloads == [] and r.cleanups == 0
    assert r.m.used_mb() == 2600


def test_a_full_size_model_evicts_everything():
    r = Rig()
    r.touch("a", "b", "big")
    assert r.m.loaded() == ["big"] and sorted(r.unloads) == ["a", "b"]


def test_exclusive_models_get_the_whole_gpu():
    r = Rig(sizes={"a": 100, "b": 100})
    r.add("solo", 100, exclusive=True)
    r.touch("a", "b")
    r.touch("solo")                       # fits by size, but exclusive: others go
    assert r.m.loaded() == ["solo"]
    r.touch("a")                          # and it goes when anything else is requested
    assert r.m.loaded() == ["a"] and "solo" in r.unloads


def test_a_model_in_use_is_never_evicted():
    r = Rig()
    with r.m.use("a"):
        with pytest.raises(ModelBusyError, match="in use"):
            r.touch("big")                # would require unloading a
        assert r.m.loaded() == ["a"]      # untouched
        with pytest.raises(ModelBusyError):
            r.m.release("a")
        with pytest.raises(ModelBusyError):
            r.m.release_all()
        r.touch("b")                      # fits alongside: allowed
    r.touch("big")                        # no longer in use: fine
    assert r.m.loaded() == ["big"]


def test_nested_use_of_the_same_model_is_counted():
    r = Rig()
    with r.m.use("a"):
        with r.m.use("a"):
            pass
        with pytest.raises(ModelBusyError):
            r.touch("big")                # still in use by the outer block
    r.touch("big")


def test_out_of_memory_triggers_unload_and_one_retry():
    r = Rig()
    r.touch("b")
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("CUDA failed with error out of memory")
        return "recovered"

    r.m.register("flaky", flaky, vram_mb=100)
    with r.m.use("flaky") as model:
        assert model == "recovered"
    assert len(attempts) == 2 and r.unloads == ["b"] and r.cleanups >= 1


def test_oom_retry_leaves_models_in_use_alone():
    r = Rig()
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("out of memory")
        return "ok"

    r.m.register("flaky", flaky, vram_mb=100)
    with r.m.use("a"):
        with r.m.use("flaky"):
            pass
        assert r.m.loaded() == ["a", "flaky"]


def test_a_second_oom_is_a_real_failure():
    r = Rig()
    r.m.register("bad", lambda: (_ for _ in ()).throw(RuntimeError("out of memory")), vram_mb=100)
    with pytest.raises(RuntimeError, match="out of memory"):
        with r.m.use("bad"):
            pass
    assert "bad" not in r.m.loaded()


def test_other_errors_are_not_retried():
    r = Rig()
    calls = []

    def broken():
        calls.append(1)
        raise ValueError("corrupt weights")

    r.m.register("broken", broken, vram_mb=100)
    with pytest.raises(ValueError, match="corrupt"):
        with r.m.use("broken"):
            pass
    assert len(calls) == 1 and r.m.loaded() == []


def test_unknown_model_lists_whats_registered():
    with pytest.raises(ModelError, match="registered: \\['a'"):
        Rig().m.use("nope").__enter__()


def test_a_failing_unloader_does_not_block_freeing_memory():
    r = Rig()
    r.m.register("fragile", lambda: "x", vram_mb=100, unloader=lambda model: 1 / 0)
    r.touch("fragile")
    r.m.release("fragile")
    assert r.m.loaded() == []


def test_release_and_release_all():
    r = Rig()
    r.touch("a", "b")
    r.m.release("a")
    assert r.m.loaded() == ["b"] and r.unloads == ["a"]
    r.m.release("a")                       # already unloaded: no-op
    r.m.release_all()
    assert r.m.loaded() == [] and sorted(r.unloads) == ["a", "b"]
    r.touch("a")                           # can be loaded again afterwards
    assert r.loads.count("a") == 2


def test_status_text():
    r = Rig()
    assert r.m.status() == "0/5000 MB declared; loaded: nothing"
    r.touch("a")
    assert "a (~2000 MB)" in r.m.status()


# ---- budget detection ------------------------------------------------------

def test_budget_comes_from_the_detected_gpu_once():
    calls = []
    m = ModelManager(gpu_info=lambda: calls.append(1) or (200, 6144), env={})
    assert m.budget_mb == int(6144 * 0.85) and m.budget_mb == 5222
    assert len(calls) == 1


def test_budget_env_override_and_no_gpu_default():
    assert ModelManager(gpu_info=lambda: (0, 6144), env={"VIDEOAGENT_VRAM_MB": "3000"}).budget_mb == 3000
    assert ModelManager(gpu_info=lambda: None, env={}).budget_mb == DEFAULT_BUDGET_MB
    assert default_budget_mb({"VIDEOAGENT_VRAM_MB": "oops"}, 4096) == int(4096 * 0.85)


@pytest.mark.parametrize("text,expected", [
    ("1024, 6144\n", (1024, 6144)),
    ("512, 8192\n300, 24576\n", (512, 8192)),          # first GPU wins
    ("", None), ("No devices were found", None), ("a, b", None),
])
def test_parse_nvidia_smi(text, expected):
    assert parse_nvidia_smi(text) == expected


@pytest.mark.parametrize("exc,expected", [
    (RuntimeError("CUDA failed with error out of memory"), True),
    (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
    (type("OutOfMemoryError", (Exception,), {})("x"), True),
    (ValueError("bad input"), False),
])
def test_is_oom(exc, expected):
    assert is_oom(exc) is expected


def test_free_gpu_memory_never_raises_and_does_not_import_torch():
    import sys

    had = "torch" in sys.modules
    free_gpu_memory()
    assert ("torch" in sys.modules) == had


def test_free_gpu_memory_empties_the_torch_cache_when_torch_is_loaded(monkeypatch):
    import sys

    emptied = []
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: emptied.append(1)))
    monkeypatch.setitem(sys.modules, "torch", fake)
    free_gpu_memory()
    assert emptied == [1]


def test_threads_share_the_manager_safely():
    r = Rig(sizes={"a": 1000, "b": 1000})
    barrier = threading.Barrier(4)

    def work(name):
        barrier.wait()
        for _ in range(20):
            with r.m.use(name):
                time.sleep(0.001)

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(work, ["a", "b", "a", "b"]))
    assert sorted(r.loads) == ["a", "b"]    # each loaded once, never evicted: they fit together


# ---- wired into Whisper and perception -------------------------------------

def test_register_whisper_declares_vram_by_size_and_is_idempotent():
    m = ModelManager(5000)
    assert register_whisper(m, "small") == "whisper:small"
    assert register_whisper(m, "small") == "whisper:small"
    assert m.is_registered("whisper:small") and WHISPER_VRAM_MB["small"] < WHISPER_VRAM_MB["medium"]


def test_perception_shares_one_loaded_whisper_across_files(monkeypatch, scene_clip, sample_clip, tmp_path):
    loaded = []

    class FakeWhisper:
        def transcribe(self, path, **kw):
            seg = SimpleNamespace(start=0.1, end=1.0, text="hello", avg_logprob=-0.1, no_speech_prob=0.0)
            return iter([seg]), SimpleNamespace(language="en")

    monkeypatch.setattr("videoagent.perception.transcribe.load_model",
                        lambda *a, **k: loaded.append(1) or FakeWhisper())
    manager = ModelManager(5000)
    cfg = PerceptionConfig(whisper_model="small")
    d1 = build_perception_doc(scene_clip, cfg, tmp_path, model_manager=manager)
    d2 = build_perception_doc(sample_clip, cfg, tmp_path, model_manager=manager)
    assert d1.transcript[0].text == "hello" and d2.transcript[0].text == "hello"
    assert len(loaded) == 1 and manager.loaded() == ["whisper:small"]
