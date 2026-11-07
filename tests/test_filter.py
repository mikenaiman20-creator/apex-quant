import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from server.ingestion.filter import GeminiUnavailable, extract_strategy
from server.ingestion.sources import ResearchDocument


def _reply():
    return {
        "title": "ATR breakout",
        "category": "stocks",
        "rules": {
            "indicators": ["ATR(14)", "volume > 1.5x average"],
            "entry_condition": "Close breaks above range by 0.1%.",
            "exit_condition": "Exit at 1.25 ATR stop or 2R target.",
            "stop_loss_atr": 1.25,
            "take_profit_mult": 2.0,
        },
        "conviction_score": 0.82,
    }


def _install_fake_genai(monkeypatch, generate_content):
    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.options = kwargs

    class Models:
        def generate_content(self, **kwargs):
            return generate_content(**kwargs)

    class Client:
        def __init__(self, api_key):
            assert api_key == "test-key"
            self.models = Models()

    google = ModuleType("google")
    google.__path__ = []
    genai = ModuleType("google.genai")
    types = ModuleType("google.genai.types")
    types.GenerateContentConfig = GenerateContentConfig
    genai.Client = Client
    genai.types = types
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types)


def test_gemini_structured_output_is_validated(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(_reply()))

    _install_fake_genai(monkeypatch, generate_content)
    document = ResearchDocument("ATR article", "https://example.org/a", "rules", "rss")
    result = extract_strategy(document)

    assert result["source_url"] == document.source_url
    assert result["storage_category"] == "stock"
    assert calls[0]["model"] == "gemini-2.5-flash"
    assert calls[0]["config"].options["response_mime_type"] == "application/json"
    assert calls[0]["config"].options["response_schema"]["required"] == [
        "title", "category", "rules", "conviction_score",
    ]


def test_missing_model_risk_numbers_are_filled_only_from_explicit_source_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    response = _reply()
    response["rules"]["stop_loss_atr"] = None
    response["rules"]["take_profit_mult"] = None
    _install_fake_genai(
        monkeypatch, lambda **_kwargs: SimpleNamespace(text=json.dumps(response)),
    )
    result = extract_strategy(ResearchDocument(
        "ATR article",
        "https://example.org/a",
        "Use ATR(14); enter above range by 0.1%; exit at a 1.25 ATR stop or 2R target.",
        "rss",
    ))
    assert result["rules"]["stop_loss_atr"] == 1.25
    assert result["rules"]["take_profit_mult"] == 2.0


@pytest.mark.parametrize(
    "wrap",
    [
        lambda value: {**value, "rules": json.dumps(value["rules"])},
        lambda value: {"strategy": value},
    ],
)
def test_extractor_normalizes_json_encoded_rules_or_strategy_wrapper(monkeypatch, wrap):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    response = wrap(_reply())
    _install_fake_genai(
        monkeypatch, lambda **_kwargs: SimpleNamespace(text=json.dumps(response)),
    )
    result = extract_strategy(ResearchDocument(
        "ATR article", "https://example.org/a", "rules", "rss",
    ))
    assert result["rules"]["stop_loss_atr"] == 1.25
    assert result["rules"]["take_profit_mult"] == 2.0


def test_extractor_keeps_rejecting_non_json_rules(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    response = _reply()
    response["rules"] = "Use ATR(14) and exit at 2R."
    _install_fake_genai(
        monkeypatch, lambda **_kwargs: SimpleNamespace(text=json.dumps(response)),
    )
    assert extract_strategy(ResearchDocument(
        "ATR article", "https://example.org/a", "rules", "rss",
    )) is None


def test_missing_risk_numbers_remain_invalid_when_source_is_not_explicit(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    response = _reply()
    response["rules"]["stop_loss_atr"] = None
    response["rules"]["take_profit_mult"] = None
    _install_fake_genai(
        monkeypatch, lambda **_kwargs: SimpleNamespace(text=json.dumps(response)),
    )
    result = extract_strategy(ResearchDocument(
        "ATR article", "https://example.org/a", "Use a volatility-based stop.", "rss",
    ))
    assert result is None


def test_gemini_failure_does_not_fall_back_to_ollama(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _install_fake_genai(
        monkeypatch,
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("service unavailable")),
    )
    monkeypatch.setattr(
        "server.ingestion.filter._extract_with_ollama",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected Ollama fallback")),
    )

    with pytest.raises(GeminiUnavailable, match="Gemini strategy extraction failed"):
        extract_strategy(
            ResearchDocument("ATR article", "https://example.org/a", "rules", "rss")
        )
