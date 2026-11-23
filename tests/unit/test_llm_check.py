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
