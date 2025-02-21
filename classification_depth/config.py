from dataclasses import dataclass
@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 8