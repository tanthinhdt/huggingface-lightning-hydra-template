from transformers import PretrainedConfig


class HFModelConfig(PretrainedConfig):
    """Configuration class for HuggingFace models. Inherits from `PretrainedConfig`, so it can be
    loaded/saved using the same methods as HuggingFace models, and is compatible with HuggingFace
    Trainer.

    You can add any additional attributes you need for your model here. They will be saved in the
    config file along with the pretrained model config attributes.
    """

    def __init__(
        self,
        encoder_pretrained_model_name_or_path: str = None,
        hidden_dim: int = None,
        encoder_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_pretrained_model_name_or_path = encoder_pretrained_model_name_or_path
        self.hidden_dim = hidden_dim
        self.encoder_dropout = encoder_dropout
