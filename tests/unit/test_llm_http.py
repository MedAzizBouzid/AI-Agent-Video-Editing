"""Real HTTP: the OpenAI SDK talks to a tiny local server, so base_url handling, headers, JSON mode,
status-code translation and the whole fallback stack are exercised without any network access."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from videoagent.llm import AuthError, RateLimitError, TransientError, build_client
from videoagent.llm.backends import OpenAIBackend
from videoagent.llm.errors import BadRequestError, ModelNotFoundError

from .llm_helpers import cfg


def completion(text="hello", usage=True, finish="stop"):
    body = {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": finish}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
    return 200, {}, body


class Server:
    def __init__(self):
        self.responses: list[tuple[int, dict, dict]] = []
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep test output clean
                pass

            def _reply(self):
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"null")
                outer.requests.append({"path": self.path, "body": body,
                                       "headers": {k.lower(): v for k, v in self.headers.items()}})
                status, headers, payload = (
                    outer.responses.pop(0) if len(outer.responses) > 1 else outer.responses[0]
                )
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            do_POST = _reply

            def do_GET(self):
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}})
                data = json.dumps({"object": "list", "data": [{"id": "zeta", "object": "model"},
                                                              {"id": "alpha", "object": "model"}]}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = Server()
    yield s
    s.close()


def backend_for(server, **kw):
    return OpenAIBackend(cfg("p", base_url=server.url, **kw), "sk-test",
                         http_client=httpx.Client(trust_env=False))


def chat(backend, **kw):
    return backend.chat([{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=None,
                        response_format=kw.get("response_format"))


def test_success_maps_text_usage_and_request(server):
    server.responses = [completion("مرحبا")]
    res = chat(backend_for(server), response_format={"type": "json_object"})
    assert (res.text, res.prompt_tokens, res.completion_tokens, res.estimated) == ("مرحبا", 11, 7, False)
    assert res.provider == "p" and res.finish_reason == "stop" and res.latency_s >= 0
    req = server.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["authorization"] == "Bearer sk-test"
    assert req["body"]["model"] == "p-model" and req["body"]["response_format"] == {"type": "json_object"}
    assert req["body"]["temperature"] == 0.1 and "max_tokens" not in req["body"]


def test_missing_usage_is_estimated(server):
    server.responses = [completion("a" * 40, usage=False)]
    res = chat(backend_for(server))
    assert res.estimated and res.completion_tokens == 10


def test_extra_headers_are_sent(server):
    server.responses = [completion()]
    chat(backend_for(server, headers={"HTTP-Referer": "https://example.com"}))
    assert server.requests[0]["headers"]["http-referer"] == "https://example.com"


@pytest.mark.parametrize("status,exc", [
    (401, AuthError), (403, AuthError), (402, AuthError), (404, ModelNotFoundError),
    (400, BadRequestError), (422, BadRequestError), (500, TransientError), (503, TransientError),
    (408, TransientError),
])
def test_status_codes_become_the_right_error(server, status, exc):
    server.responses = [(status, {}, {"error": {"message": "nope", "type": "x"}})]
    with pytest.raises(exc):
        chat(backend_for(server))


def test_rate_limit_carries_retry_after(server):
    server.responses = [(429, {"Retry-After": "7"}, {"error": {"message": "slow down"}})]
    with pytest.raises(RateLimitError) as err:
        chat(backend_for(server))
    assert err.value.retry_after == 7.0


def test_rate_limit_without_header_has_no_retry_after(server):
    server.responses = [(429, {}, {"error": {"message": "slow down"}})]
    with pytest.raises(RateLimitError) as err:
        chat(backend_for(server))
    assert err.value.retry_after is None


def test_empty_content_is_transient(server):
    server.responses = [completion("   ")]
    with pytest.raises(TransientError, match="empty"):
        chat(backend_for(server))


def test_connection_refused_is_transient(server):
    backend = backend_for(server)
    server.close()
    with pytest.raises(TransientError):
        chat(backend)


def test_list_models_sorted(server):
    assert backend_for(server).list_models() == ["alpha", "zeta"]


def test_full_stack_over_http_falls_back_and_validates(server, tmp_path):
    """Provider 1 returns 429, provider 2 returns valid JSON: complete_json gets a typed object."""
    from pydantic import BaseModel

    class Out(BaseModel):
        ok: bool

    limited, healthy = Server(), Server()
    try:
        limited.responses = [(429, {"Retry-After": "30"}, {"error": {"message": "quota"}})]
        healthy.responses = [completion('{"ok": true}')]
        conf = tmp_path / "p.yaml"
        conf.write_text(f"""
providers:
  - {{name: limited, base_url: "{limited.url}", api_key_env: K1, model: m}}
  - {{name: healthy, base_url: "{healthy.url}", api_key_env: K2, model: m,
      price_in_per_mtok: 1.0, price_out_per_mtok: 2.0}}
""")
        client = build_client(
            conf, env={"K1": "a", "K2": "b"}, load_dotenv_file=False,
            backend_factory=lambda c, key: OpenAIBackend(c, key, http_client=httpx.Client(trust_env=False)),
        )
        assert client.complete_json("go", Out, tag="t") == Out(ok=True)
        assert client.chain.status()["limited"].startswith("cooling down")
        assert client.usage.by("provider")["healthy"]["calls"] == 1
        assert client.usage.cost_usd == pytest.approx((11 * 1.0 + 7 * 2.0) / 1e6)
        # JSON mode was requested on the wire
        assert healthy.requests[0]["body"]["response_format"] == {"type": "json_object"}
    finally:
        limited.close()
        healthy.close()
