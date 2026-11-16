import pytest
from pydantic import ValidationError

from videoagent.schemas import EDL, BriefSpec, Clip


def test_clip_rejects_reversed_times():
    with pytest.raises(ValidationError):
        Clip(source="a.mp4", start=5, end=2)


def test_edl_total_duration_respects_speed():
    edl = EDL(
        platform="shorts",
        title="t",
        clips=[Clip(source="a.mp4", start=0, end=10, speed=2.0)],
    )
    assert edl.total_duration == 5.0


def test_brief_requires_inputs():
    with pytest.raises(ValidationError):
        BriefSpec(inputs=[], goal="x")
