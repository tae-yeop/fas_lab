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
    json_path:str = '/purestorage/project/tyk/3_CUProjects/iBeta/dataset/landmark_label/TEST/labels.json'
    wandb_key:str = 'local-73177de041f41c769eb8cbdccb982a9a5406fab7'
    wandb_host:str = 'http://wandb.artfacestudio.com'
    wandb_project_name:str = 'test'
    wandb_run_name:str = 'test'