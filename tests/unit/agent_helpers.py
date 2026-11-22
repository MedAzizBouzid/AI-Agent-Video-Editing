"""Fakes for the agent tests: a real LLMClient over a scripted backend, with fake perception,
rendering and critique, so the graph's decisions can be tested without video or network."""

from __future__ import annotations

import json
from pathlib import Path

from videoagent.agent import AgentConfig, Deps
from videoagent.config import load_platforms
from videoagent.llm.client import LLMClient
from videoagent.schemas import BriefSpec, CriticReport, Issue, PerceptionDoc, SourceInfo, TranscriptSegment

from .llm_helpers import FakeBackend, make_chain, ok

PLATFORMS = load_platforms()
SPEECH = [(2.0, 6.0, "first thing the client says"), (10.0, 14.0, "second thing worth keeping"),
          (30.0, 34.0, "third thing to cut")]


def edl_json(clips=((2, 6), (10, 14)), *, source="raw.mp4", platform="shorts", title="Demo", **extra) -> str:
    body = {"platform": platform, "title": title,
            "clips": [{"source": source, "start": s, "end": e} for s, e in clips], **extra}
    return json.dumps(body)


def copy_json(title="Great title", hashtags=("a", "b")) -> str:
    return json.dumps({"title": title, "description": "Short description.", "hashtags": list(hashtags)})


def intake_json(**kw) -> str:
    return json.dumps({"goal": "30s teaser", **kw})


def report(verdict="pass", score=0.9, issues=()) -> CriticReport:
    return CriticReport(verdict=verdict, score=score, summary=f"{verdict} summary", issues=list(issues))


def major(description="the hook is weak", at=1.0, fix="start with the result") -> Issue:
    return Issue(severity="major", timestamp_s=at, description=description, suggested_fix=fix, source="llm")


def make_doc(name="raw.mp4", duration=60.0, speech=SPEECH) -> PerceptionDoc:
    return PerceptionDoc(
        source=SourceInfo(path=name, sha256="x", duration_s=duration, width=1920, height=1080, fps=30),
        transcript=[TranscriptSegment(start=s, end=e, text=t) for s, e, t in speech], language="en")


class Harness:
    """Holds the fake backend and records what the fakes were asked, so tests can assert on it."""

    def __init__(self, tmp_path: Path, replies, critiques=(), *, render_errors=(), doc=None, vision=False):
        self.backend = FakeBackend(*replies)
        chain, _ = make_chain(("main", self.backend, {"vision": vision}))
        self.client = LLMClient(chain)
        self.critiques = list(critiques)
        self.render_errors = list(render_errors)
        self.doc = doc or make_doc()
        self.renders: list = []
        self.critique_calls: list = []
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.out = tmp_path / "out"
        self.input = tmp_path / "raw.mp4"
        self.input.write_bytes(b"fake video")
        self.deps = Deps(
            client=self.client, platforms=PLATFORMS, perceive=self._perceive,
            render=self._render, critique=self._critique, voiceover_available=False)

    def _perceive(self, path, name):
        return self.doc.model_copy(update={"source": self.doc.source.model_copy(update={"path": name})})

    def _render(self, edl, dst, brief):
        n = len(self.renders) + 1
        self.renders.append(edl)
        if self.render_errors and (err := self.render_errors.pop(0)):
            raise err
        Path(dst).write_bytes(f"render {n}".encode())
        return Path(dst)

    def _critique(self, video, edl, brief, docs, platform):
        self.critique_calls.append((video, edl, platform))
        return self.critiques.pop(0) if len(self.critiques) > 1 else self.critiques[0]

    def messages(self, call: int) -> str:
        """All text the LLM was sent on backend call number `call` (0-based)."""
        return "\n".join(str(m["content"]) for m in self.backend.calls[call]["messages"])

    def brief(self, **kw) -> BriefSpec:
        return BriefSpec(inputs=[str(self.input)], goal="30s teaser", platforms=["shorts"], **kw)

    def run(self, cfg=None, **kw):
        from videoagent.agent import run_pipeline

        kw.setdefault("brief", self.brief())
        return run_pipeline([self.input], self.deps, cfg or AgentConfig(), out_dir=self.out, **kw)


def replies(*texts):
    return [ok(t) for t in texts]
