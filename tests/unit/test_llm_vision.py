import httpx
import pytest

from videoagent.llm.errors import AllProvidersFailed
from videoagent.llm.images import content_size, image_part, user_message_with_images
from videoagent.llm.prompts import load_prompt

from .llm_helpers import MSG, FakeBackend, make_chain, ok
from .test_llm_http import Server, backend_for, completion


@pytest.fixture
def jpg(tmp_path):
    p = tmp_path / "f.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    return p


def test_image_part_is_a_base64_data_url(jpg):
    part = image_part(jpg)
    assert part["type"] == "image_url" and part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_message_builder_orders_text_captions_images(jpg):
    msg = user_message_with_images("look", [jpg, jpg], ["Frame 1:", "Frame 2:"])
    kinds = [(p["type"], p.get("text")) for p in msg["content"]]
    assert kinds == [("text", "look"), ("text", "Frame 1:"), ("image_url", None),
                     ("text", "Frame 2:"), ("image_url", None)]
    assert msg["role"] == "user"


def test_content_size_counts_text_and_images(jpg):
    assert content_size("hello") == (5, 0)
    assert content_size(user_message_with_images("abc", [jpg])["content"]) == (3, 1)


def test_load_prompt():
    assert "quality reviewer" in load_prompt("critic")
    with pytest.raises(FileNotFoundError):
        load_prompt("nope")


def test_image_requests_only_go_to_vision_providers():
    text_only, eyes = FakeBackend(ok("text")), FakeBackend(ok("eyes"))
    chain, _ = make_chain(("text", text_only), ("eyes", eyes, {"vision": True}))
    assert chain.chat(MSG, needs_vision=True).text == "eyes"
    assert text_only.calls == []
    assert chain.chat(MSG).text == "text"                      # ordinary calls keep their order


def test_no_vision_provider_explains_how_to_fix_it():
    chain, _ = make_chain(("a", FakeBackend(ok())))
    assert not chain.supports_vision()
    with pytest.raises(AllProvidersFailed) as exc:
        chain.chat(MSG, needs_vision=True)
    assert "vision: true" in exc.value.failures["a"]


def test_supports_vision_ignores_disabled_providers():
    chain, _ = make_chain(("eyes", FakeBackend(ok()), {"vision": True}))
    assert chain.supports_vision()
    chain.providers[0].disabled = "no key"
    assert not chain.supports_vision()


def test_images_reach_the_wire_and_count_toward_estimated_tokens(jpg):
    s = Server()
    try:
        s.responses = [completion("a" * 40, usage=False)]
        msg = user_message_with_images("describe", [jpg])
        res = backend_for(s, vision=True).chat([msg], temperature=0.1, max_tokens=None, response_format=None)
        sent = s.requests[0]["body"]["messages"][0]["content"]
        assert [p["type"] for p in sent] == ["text", "image_url"]
        assert res.estimated and res.prompt_tokens >= 700       # the image is not free
    finally:
        s.close()
