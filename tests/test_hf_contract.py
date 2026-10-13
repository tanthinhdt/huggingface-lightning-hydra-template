"""Check the original sequence-model path without downloading an encoder or dataset."""

from types import SimpleNamespace

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from src.models.components.hfmodel.configuration import HFModelConfig
from src.models.components.hfmodel.modelling import HFModelForTask
from src.models.components.metrics import Metrics


def test_sequence_model_contract(monkeypatch):
    class Encoder(nn.Module):
        def forward(self, input_ids, attention_mask, output_attentions=False):
            hidden = torch.ones(*input_ids.shape, 4)
            attention = (torch.ones(len(input_ids), 1, input_ids.shape[1], input_ids.shape[1]),)
            return BaseModelOutput(
                last_hidden_state=hidden, attentions=attention if output_attentions else None
            )

    monkeypatch.setattr(
        "src.models.components.hfmodel.modelling.AutoModel.from_pretrained",
        lambda *args, **kwargs: Encoder(),
    )
    config = HFModelConfig(
        encoder_pretrained_model_name_or_path="offline", hidden_dim=4, num_labels=2
    )
    model = HFModelForTask(config)
    output = model(
        torch.ones(2, 6, dtype=torch.long),
        torch.ones(2, 6),
        labels=torch.tensor([0, 1]),
        output_attentions=True,
    )
    assert output.logits.shape == (2, 2)
    assert output.loss.ndim == 0 and torch.isfinite(output.loss)
    assert len(output.attentions) == 1
    metrics = Metrics(2)
    metrics.update(output.logits.argmax(dim=-1), torch.tensor([0, 1]))
    assert "mcc" in metrics.compute()
