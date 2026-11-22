import json
import shutil
from pathlib import Path

import pytest

from apps import cli
from videoagent.console import use_utf8_console

from .agent_helpers import Harness, copy_json, edl_json, intake_json, major, replies, report


def run_cli(tmp_path, texts, critiques=(report(),), extra=(), **kw):
    t = Harness(tmp_path, replies(*texts), critiques, **kw)
    argv = ["run", str(t.input), "--brief", "تيزر 30 ثانية", "--out", str(t.out), *extra]
    code = cli.main(argv, deps_factory=lambda cfg: t.deps)
    return t, code


def test_a_good_run_exits_zero_and_prints_the_arabic_title(tmp_path, capsys):
    t, code = run_cli(tmp_path, [intake_json(language="ar"), edl_json(), copy_json("عرض خاص")])
    out = capsys.readouterr().out
    assert code == 0 and "=== shorts: READY ===" in out and "عرض خاص" in out
    assert (t.out / "raw.shorts.mp4").is_file()
    assert "calls" in out.splitlines()[-1]                      # the closing line reports the LLM usage


def test_flags_override_the_llm_reading_of_the_brief(tmp_path):
    t, code = run_cli(tmp_path, [intake_json(platforms=["youtube"], language="fr"), edl_json(platform="tiktok"),
                                 copy_json()], extra=["--platform", "tiktok", "--language", "ar-TN"])
    assert code == 0 and (t.out / "raw.tiktok.mp4").is_file() and not (t.out / "raw.youtube.mp4").exists()
    assert "PLATFORM: tiktok" in t.messages(1) and '"language": "ar-TN"' in t.messages(1)


def test_exit_codes_distinguish_review_and_failure(tmp_path):
    _, review = run_cli(tmp_path / "a", [intake_json(), edl_json()],
                        critiques=[report("revise", 0.5, [major()])], extra=["--max-revisions", "0"])
    _, failed = run_cli(tmp_path / "b", [intake_json(), edl_json(clips=[(2, 90)])], extra=[])
    assert (review, failed) == (1, 2)


def test_plan_only_writes_the_plan_and_no_video(tmp_path, capsys):
    t, code = run_cli(tmp_path, [intake_json(), edl_json()], extra=["--plan-only"])
    assert code == 0 and not list(t.out.glob("*.mp4")) and (t.out / "raw.shorts.edl.json").is_file()
    assert "PLANNED" in capsys.readouterr().out


def test_brief_can_be_read_from_a_utf8_file(tmp_path):
    f = tmp_path / "brief.txt"
    f.write_text("فيديو للمنتج الجديد", encoding="utf-8")
    assert cli.read_brief(f"@{f}") == "فيديو للمنتج الجديد" and cli.read_brief("plain") == "plain"


def test_a_missing_input_file_is_reported_without_a_traceback(tmp_path, capsys):
    code = cli.main(["run", str(tmp_path / "nope.mp4"), "--brief", "x"], deps_factory=lambda c: None)
    assert code == 2 and "input file not found" in capsys.readouterr().err


def test_console_is_switched_to_utf8_so_arabic_cannot_crash_print(monkeypatch):
    calls = []

    class Stream:
        def reconfigure(self, **kw):
            calls.append(kw)

    monkeypatch.setattr("sys.stdout", Stream())
    monkeypatch.setattr("sys.stderr", object())          # no reconfigure(): must be tolerated
    use_utf8_console()
    assert calls == [{"encoding": "utf-8", "errors": "replace"}]
