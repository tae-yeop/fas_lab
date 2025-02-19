from dataclasses import dataclass

@dataclass
class TraningConfig:
    batch_size:int = 32
    lr:float = 0.001
    epochs:int = 10
    model:str = 'simple'
    pl_strategy:str ='deepspeed_stage_2'
    pl_precision:str ='bf16-mixed'
    seed:int = 3
    json_path: