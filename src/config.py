from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "t5-small"
    output_dir: str = "model/checkpoints/best-model"
    train_batch_size: int = 8
    eval_batch_size: int = 8
    epochs: int = 3
    lr: float = 3e-4
    max_input_length: int = 128
    max_target_length: int = 64
    seed: int = 42
    device: str = "cuda" if __import__('torch').cuda.is_available() else "cpu"

config = Config()
