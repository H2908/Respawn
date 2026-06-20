import pytest

from respawn.llm import complete


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        complete([], model="some-model", provider="unknown_provider")


def test_anthropic_import_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("no anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="pip install anthropic"):
        complete([{"role": "user", "content": "hi"}], model="claude-sonnet-4-6")


def test_openai_import_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="pip install openai"):
        complete([{"role": "user", "content": "hi"}], model="gpt-4o")
