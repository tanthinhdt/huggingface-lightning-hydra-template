import torch
from torch import nn
from typing import Optional, Tuple, Dict, Union, List
from transformers import PreTrainedModel, AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from .configuration import HFModelConfig


def trim_special_tokens(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Trim special tokens ([CLS], [SEP]) from last_hidden_state and attention_mask.

    Parameters
    ----------
    last_hidden_state : torch.Tensor
        The last hidden state from the encoder (batch_size, seq_len, hidden_dim).
    attention_mask : torch.Tensor
        The attention mask for the input sequences (batch_size, seq_len).
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Trimmed last_hidden_state and attention_mask with special tokens removed.
    """
    seq_lengths = attention_mask.sum(dim=1)  # (batch_size,)

    batch_size, total_len, hidden_dim = last_hidden_state.shape
    trimmed_embeddings = []
    trimmed_attention_masks = []

    for i in range(batch_size):
        seq_len = seq_lengths[i].item()
        # Extract tokens between [CLS] and [SEP]: positions 1 to seq_len-2
        embeddings = last_hidden_state[i, 1:seq_len-1, :]  # Exclude [CLS] and [SEP]
        mask = attention_mask[i, 1:seq_len-1]  # Corresponding mask

        # Pad to consistent length if needed
        pad_len = total_len - seq_len
        if pad_len > 0:
            embeddings = torch.cat([embeddings, torch.zeros(pad_len, hidden_dim, device=embeddings.device)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad_len, device=mask.device)], dim=0)
        
        trimmed_embeddings.append(embeddings)
        trimmed_attention_masks.append(mask)

    trimmed_embeddings = torch.stack(trimmed_embeddings)  # (batch_size, total_len, hidden_dim)
    trimmed_attention_masks = torch.stack(trimmed_attention_masks)  # (batch_size, total_len)
    return trimmed_embeddings, trimmed_attention_masks


class LabelEncoder:
    """Utility class for encoding and decoding labels."""

    def __init__(
        self,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
    ):
        """Initialize the LabelEncoder with label-to-id and id-to-label mappings.
        
        Parameters
        ----------
        label2id : Dict[str, int], optional
            A dictionary mapping label names to integer IDs.
        id2label : Dict[int, str], optional
            A dictionary mapping integer IDs back to label names.
        """
        self.label2id = label2id
        self.id2label = id2label

    def __call__(self, labels: Union[List[str], str]) -> Union[List[int], int]:
        """Encode labels into their corresponding integer IDs.
        
        Parameters
        ----------
        labels : Union[List[str], str]
            The label(s) to be encoded.
            
        Returns
        -------
        Union[List[int], int]
            The encoded integer ID(s).
        """
        if self.label2id is None:
            return labels
        if isinstance(labels, str):
            return int(self.label2id[labels])
        return [int(self.label2id[label]) for label in labels]

    def decode(self, ids: Union[List[int], int]) -> Union[List[str], str]:
        """Decode integer IDs back into their corresponding label names.
        
        Parameters
        ----------
        ids : Union[List[int], int]
            The integer ID(s) to be decoded.

        Returns
        -------
        Union[List[str], str]
            The decoded label name(s).
        """
        if self.id2label is None:
            return ids
        if isinstance(ids, int):
            return str(self.id2label[ids])
        return [str(self.id2label[id]) for id in ids]


class HFModelProcessor:
    """Processor for the MESP model, responsible for tokenization and label encoding."""

    def __init__(self, config: HFModelConfig):
        """Initialize the HFModelProcessor with the given configuration."""
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder_pretrained_model_name_or_path)
        self.label_encoder = LabelEncoder(config.sp2id, config.id2sp)

    @property
    def num_labels(self) -> int:
        """Get the number of labels for classification tasks.
        
        Returns
        -------
        int
            The number of labels for classification tasks.
        """
        return self.config.num_labels

    def tokenize_sequences(self, sequences: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize input sequences using the tokenizer.

        Parameters
        ----------
        sequences : List[str]
            A list of input sequences to be tokenized.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing tokenized inputs, including 'input_ids' and 'attention_mask'.
        """
        return self.tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )

    def encode_labels(self, labels: List[str]) -> torch.Tensor:
        """Encode labels into integer IDs using the label encoder.

        Parameters
        ----------
        labels : List[str]
            A list of label names to be encoded.

        Returns
        -------
        torch.Tensor
            A tensor containing the encoded integer IDs for the labels.
        """
        return torch.tensor(self.label_encoder(labels), dtype=torch.long)


class HFModelClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_labels: int,
    ):
        """Initialize the classification head.
        
        Parameters
        ----------
        hidden_dim : int
            The hidden dimension of the input features.
        num_layers : int
            The number of Mamba layers in the block.
        num_labels : int
            The number of output labels for classification.
        """
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_labels)
    
    def forward(
        self,
        encoder_last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """Compute forward pass for the token classification head.

        Parameters
        ----------
        encoder_last_hidden_state : torch.Tensor
            The last hidden state from the encoder (batch_size, seq_len, hidden_dim).
        attention_mask : torch.Tensor
            The attention mask for the input sequences (batch_size, seq_len).
        output_attentions : bool, optional
            Whether to return attention weights from the model.
        
        Returns
        -------
        SequenceClassifierOutput
            Model output containing logits, last hidden states, and attentions
            from the token classification head.
        """
        context, feature_attention_masks = trim_special_tokens(
            last_hidden_state=encoder_last_hidden_state,
            attention_mask=attention_mask,
        )

        logits = self.classifier(context)

        attentions = None
        if output_attentions:
            attentions = attentions

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=context,
            attentions=attentions,
        )


class HFModel(PreTrainedModel):
    config_class = HFModelConfig

    def __init__(self, config: HFModelConfig):
        super().__init__(config)
        self.config = config
        self.encoder = AutoModel.from_pretrained(
            config.encoder_pretrained_model_name_or_path,
            ignore_mismatched_sizes=True,
            add_pooling_layer=False,
        )
        self.encoder_dropout = nn.Dropout(config.encoder_dropout)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """Forward pass of the model. You need to implement this method for your model."""
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        encoder_last_hidden_state = self.encoder_dropout(encoder_outputs.last_hidden_state)

        return SequenceClassifierOutput(
            last_hidden_state=encoder_last_hidden_state,
            attentions=encoder_outputs.attentions,
        )

    def get_processor(self):
        """Get the processor for the model. This can be used to preprocess the data in the datamodule."""
        return HFModelProcessor(self.config)


class HFModelForTask(HFModel):

    def __init__(self, config: HFModelConfig):
        """Initialize the HFModelForTask model.

        Parameters
        ----------
        config : HFModelConfig
            Configuration object containing model hyperparameters and settings.
        """
        super().__init__(config)
        self.classifier = HFModelClassifier(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_labels=config.num_labels,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """Compute forward pass for signal peptide detection.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tokenized input sequences (batch_size, seq_len).
        attention_mask : torch.Tensor
            Attention mask for input sequences (batch_size, seq_len).
        labels : torch.Tensor, optional
            Ground truth labels for both tasks (batch_size, 2) with [spd_label, spc_label].
        output_attentions : bool, optional
            Whether to return attention weights from the model.
        
        Returns
        -------
        SequenceClassifierOutput
            Model output containing loss, logits, predictions, last hidden states, and attentions.
        """
        encoder_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        classifier_outputs = self.classifier(encoder_outputs.last_hidden_state, attention_mask)
        logits = classifier_outputs.logits

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        hidden_states = encoder_outputs.hidden_states + classifier_outputs.hidden_states

        attentions = None
        if output_attentions:
            attentions = encoder_outputs.attentions + classifier_outputs.attentions

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )
