"""TDD for hrf_llm.client — the OpenAI-compatible client factory."""

from hrf_llm.client import get_llm_client
from hrf_shared.config import Settings


def test_client_uses_configured_base_url_model_key():
    settings = Settings(
        llm_base_url="http://example.test:9999/v1",
        llm_api_key="secret-key",
        llm_model="my-model",
    )
    client = get_llm_client(settings)
    assert "example.test:9999" in str(client.base_url)
    assert client.api_key == "secret-key"
