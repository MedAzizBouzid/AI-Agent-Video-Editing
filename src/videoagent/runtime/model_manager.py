"""Keeps GPU models within the VRAM budget of a small card (e.g. 6 GB on an RTX 3050).

Whisper, a TTS model and anything else heavy each *declare* roughly how much VRAM they need. The
manager loads a model on first use, keeps it resident for reuse (loading is slow), and when a new
model would not fit it evicts the least-recently-used ones first. Things it guarantees:

  * a model that is currently in use is never evicted (that would crash the user mid-call);
    asking for room that only an in-use model could give raises ModelBusyError instead
  * an "exclusive" model gets the whole GPU: everything else is unloaded first, and it is
    unloaded when anything else is requested
  * if a load fails with an out-of-memory error, everything evictable is unloaded, caches are
    cleared and the load is retried once; this covers other programs also using the GPU
  * declared sizes are estimates: the budget protects against the common case (Whisper + TTS)
    and the OOM retry handles the rest

Usage:
    manager = ModelManager()
    manager.register("whisper:small", lambda: load_model("small"), vram_mb=1500)
    with manager.use("whisper:small") as model:
        model.transcribe(...)
"""

from __future__ import annotations

import gc
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("videoagent.runtime")
DEFAULT_BUDGET_MB = 5000  # used when no GPU can be detected and VIDEOAGENT_VRAM_MB is not set
BUDGET_FRACTION = 0.85  # share of total VRAM we plan to use; the rest is driver/desktop overhead


class ModelError(RuntimeError):
    pass


class ModelBusyError(ModelError):
    """Room is needed but the only models that could be evicted are in use."""


def free_gpu_memory() -> None:
    """Drop Python references and return cached GPU memory. Only touches torch if it is already
    imported (importing it just to clean up would itself cost seconds and memory)."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must never raise
            log.debug("torch cache cleanup failed", exc_info=True)


def parse_nvidia_smi(output: str) -> tuple[int, int] | None:
    """'used, total' in MiB from `nvidia-smi --query-gpu=memory.used,memory.total ...` (first GPU)."""
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            return int(parts[0]), int(parts[1])
    return None


def read_gpu_memory() -> tuple[int, int] | None:
    """(used_mb, total_mb) of the first NVIDIA GPU, or None if there is none or it cannot be read."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, errors="replace", timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_nvidia_smi(proc.stdout) if proc.returncode == 0 else None


def default_budget_mb(env: Mapping[str, str], total_mb: int | None) -> int:
    override = env.get("VIDEOAGENT_VRAM_MB")
    if override and override.isdigit():
        return int(override)
    return int(total_mb * BUDGET_FRACTION) if total_mb else DEFAULT_BUDGET_MB


def is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "out of memory" in text or "outofmemory" in text


@dataclass
class _Entry:
    name: str
    loader: Callable[[], Any]
    unloader: Callable[[Any], None] | None
    vram_mb: int
    exclusive: bool
    model: Any = None
    loaded: bool = False
    in_use: int = 0
    last_used: float = 0.0


class ModelManager:
    def __init__(
        self,
        budget_mb: int | None = None,
        *,
        cleanup: Callable[[], None] = free_gpu_memory,
        clock: Callable[[], float] = time.monotonic,
        gpu_info: Callable[[], tuple[int, int] | None] = read_gpu_memory,
        env: Mapping[str, str] | None = None,
    ):
        self._budget = budget_mb
        self._cleanup, self._clock, self._gpu_info = cleanup, clock, gpu_info
        self._env = os.environ if env is None else env
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.RLock()

    # ---- configuration --------------------------------------------------------------------

    @property
    def budget_mb(self) -> int:
        if self._budget is None:
            gpu = self._gpu_info()
            self._budget = default_budget_mb(self._env, gpu[1] if gpu else None)
            log.info("VRAM budget: %d MB", self._budget)
        return self._budget

    def register(
        self,
        name: str,
        loader: Callable[[], Any],
        *,
        vram_mb: int,
        unloader: Callable[[Any], None] | None = None,
        exclusive: bool = False,
    ) -> None:
        """Declare a model. Registering a name again is a no-op, so callers can register freely."""
        with self._lock:
            if name in self._entries:
                return
            if vram_mb > self.budget_mb:
                raise ValueError(
                    f"model '{name}' needs ~{vram_mb} MB but the VRAM budget is {self.budget_mb} MB; "
                    "use a smaller model, or raise VIDEOAGENT_VRAM_MB if the card really has room"
                )
            self._entries[name] = _Entry(name, loader, unloader, vram_mb, exclusive)

    def is_registered(self, name: str) -> bool:
        return name in self._entries

    # ---- use ------------------------------------------------------------------------------

    @contextmanager
    def use(self, name: str) -> Iterator[Any]:
        """Yield the loaded model, loading it (and evicting others) if needed. While the block runs
        the model cannot be evicted."""
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise ModelError(f"unknown model '{name}'; registered: {sorted(self._entries)}")
            if not entry.loaded:
                self._make_room(entry)
                self._load(entry)
            entry.in_use += 1
            entry.last_used = self._clock()
        try:
            yield entry.model
        finally:
            with self._lock:
                entry.in_use -= 1
                entry.last_used = self._clock()

    def release(self, name: str) -> None:
        """Unload one model now (e.g. the stage that needed it is finished)."""
        with self._lock:
            entry = self._entries.get(name)
            if entry and entry.loaded:
                if entry.in_use:
                    raise ModelBusyError(f"cannot release '{name}': it is in use")
                self._unload(entry)
                self._cleanup()

    def release_all(self) -> None:
        with self._lock:
            busy = [e.name for e in self._entries.values() if e.loaded and e.in_use]
            if busy:
                raise ModelBusyError(f"cannot release while in use: {busy}")
            for entry in self._entries.values():
                if entry.loaded:
                    self._unload(entry)
            self._cleanup()

    # ---- inspection -----------------------------------------------------------------------

    def loaded(self) -> list[str]:
        with self._lock:
            return [e.name for e in self._entries.values() if e.loaded]

    def used_mb(self) -> int:
        """Declared VRAM of everything currently loaded."""
        with self._lock:
            return sum(e.vram_mb for e in self._entries.values() if e.loaded)

    def status(self) -> str:
        loaded = ", ".join(f"{n} (~{self._entries[n].vram_mb} MB)" for n in self.loaded()) or "nothing"
        return f"{self.used_mb()}/{self.budget_mb} MB declared; loaded: {loaded}"

    # ---- internals ------------------------------------------------------------------------

    def _make_room(self, new: _Entry) -> None:
        others = [e for e in self._entries.values() if e.loaded and e is not new]
        if new.exclusive or any(e.exclusive for e in others):
            victims = others
        else:
            victims, used = [], sum(e.vram_mb for e in others)
            for e in sorted(others, key=lambda e: e.last_used):  # least recently used first
                if used + new.vram_mb <= self.budget_mb:
                    break
                victims.append(e)
                used -= e.vram_mb
        busy = [e.name for e in victims if e.in_use]
        if busy:
            raise ModelBusyError(
                f"cannot load '{new.name}' (~{new.vram_mb} MB): {busy} would have to be unloaded "
                f"but is in use ({self.status()})"
            )
        for e in victims:
            log.info("unloading %s to make room for %s", e.name, new.name)
            self._unload(e)
        if victims:
            self._cleanup()

    def _load(self, entry: _Entry) -> None:
        try:
            model = entry.loader()
        except Exception as exc:  # noqa: BLE001
            if not is_oom(exc):
                raise
            log.warning("out of memory loading %s; unloading everything else and retrying", entry.name)
            for other in self._entries.values():
                if other.loaded and other is not entry and not other.in_use:
                    self._unload(other)
            self._cleanup()
            model = entry.loader()  # a second failure is real: let it propagate
        entry.model, entry.loaded = model, True
        log.info("loaded %s (%s)", entry.name, self.status())

    def _unload(self, entry: _Entry) -> None:
        try:
            if entry.unloader is not None and entry.model is not None:
                entry.unloader(entry.model)
        except Exception:  # noqa: BLE001 - a failing unloader must not block freeing memory
            log.warning("unloader for %s failed", entry.name, exc_info=True)
        entry.model, entry.loaded = None, False
