from types import SimpleNamespace

from apps.llm_check import listing_backend

from .llm_helpers import cfg


def prov(**kw):
    return SimpleNamespace(cfg=cfg("p", api_key_env="P_KEY", model="REPLACE_WITH_X", **kw))


def test_models_can_be_listed_with_a_key_even_while_the_model_id_is_a_placeholder():
    assert listing_backend(prov(), {"P_KEY": "secret"}) is not None


def test_no_key_means_no_listing_backend():
    assert listing_backend(prov(), {}) is None
    assert listing_backend(prov(), {"P_KEY": ""}) is None


def test_describe_env_reports_keys_without_printing_secrets(tmp_path, capsys):
    from apps.llm_check import describe_env

    f = tmp_path / ".env"
    f.write_text("OPENROUTER_API_KEY=sk-secret\n", encoding="utf-8")
    lines = describe_env({"OPENROUTER_API_KEY": "sk-secret"}, f)
    out = "\n".join(lines)
    assert str(f) in out and "OPENROUTER_API_KEY: set" in out and "GEMINI_API_KEY: EMPTY" in out
    assert "sk-secret" not in out


def test_describe_env_warns_about_a_missing_file_and_a_utf16_file(tmp_path):
    from apps.llm_check import describe_env

    assert "NOT FOUND" in "\n".join(describe_env({}, None, ))
    bad = tmp_path / ".env"
    bad.write_bytes("OPENROUTER_API_KEY=x\n".encode("utf-16"))
    assert "UTF-16" in "\n".join(describe_env({}, bad))


def test_a_bom_does_not_hide_the_first_key(tmp_path, monkeypatch):
    from dotenv import dotenv_values

    f = tmp_path / ".env"
    f.write_bytes(b"\xef\xbb\xbfOPENROUTER_API_KEY=abc\n")
    assert dotenv_values(f, encoding="utf-8-sig") == {"OPENROUTER_API_KEY": "abc"}
