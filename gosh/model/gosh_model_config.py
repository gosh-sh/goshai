from transformers import GPTBigCodeConfig

class GoshModelConfig(GPTBigCodeConfig):
    model_type = "gosh_model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
