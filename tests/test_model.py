import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
model = pytest.importorskip("doom_bert.model")


def test_device_selection_and_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert model.select_device() == "mps"
    assert model.select_device("cpu") == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert model.select_device() == "cpu"
    with pytest.raises(ValueError, match="unavailable"):
        model.select_device("mps")


def test_base_model_is_not_accepted_as_a_trained_policy(monkeypatch):
    monkeypatch.setattr(
        model.AutoConfig, "from_pretrained", lambda _: transformers.ModernBertConfig()
    )
    with pytest.raises(ValueError, match="fine-tuned seven-button"):
        model.ModernBertPolicy("base-model", device="cpu")
