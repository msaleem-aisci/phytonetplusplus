import os
import yaml
from dataclasses import dataclass
from typing import List

# 1. Imports first. Then resolve project root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))  # One level up to phytonetplusplus/

@dataclass
class BasePathsConfig:
    working_dir: str
    checkpoint_dir: str
    output_dir: str

@dataclass
class DatasetConfig:
    data_dir: str
    image_size: int
    batch_size: int
    num_workers: int
    val_split: float

@dataclass
class TrainingConfig:
    epochs: int
    lr: float
    weight_decay: float
    accum_steps: int
    patience: int

@dataclass
class CheckpointConfig:
    input_ckpt: str
    working_ckpt: str
    best_model: str
    input_csv: str
    working_csv: str

@dataclass
class InferenceConfig:
    image_path: str
    model_path: str
    output_dir: str
    class_names: List[str]

@dataclass
class Config:
    base_paths: BasePathsConfig
    dataset: DatasetConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig
    inference: InferenceConfig

def load_config(config_filename: str) -> Config:
    # 2. Use the root to find the YAML file safely
    absolute_config_path = os.path.join(_PROJECT_ROOT, config_filename)
    
    with open(absolute_config_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
        
    base = raw_cfg['base_paths']
    
    # 3. Prepend project root to the base directories
    ckpt_dir = os.path.join(_PROJECT_ROOT, base['checkpoint_dir'])
    out_dir = os.path.join(_PROJECT_ROOT, base['output_dir'])
    
    # Dynamically assemble paths
    ckpt_cfg = raw_cfg['checkpoint']
    if ckpt_cfg['working_ckpt'].lower() != "none":
        ckpt_cfg['working_ckpt'] = os.path.join(ckpt_dir, ckpt_cfg['working_ckpt'])
    
    ckpt_cfg['best_model'] = os.path.join(ckpt_dir, ckpt_cfg['best_model'])
    ckpt_cfg['working_csv'] = os.path.join(ckpt_dir, ckpt_cfg['working_csv'])
    
    inf_cfg = raw_cfg['inference']
    inf_cfg['model_path'] = os.path.join(ckpt_dir, inf_cfg['model_path'])
    
    # Assign absolute output directory from base_paths
    inf_cfg['output_dir'] = out_dir

    return Config(
        base_paths=BasePathsConfig(**base),
        dataset=DatasetConfig(**raw_cfg['dataset']),
        training=TrainingConfig(**raw_cfg['training']),
        checkpoint=CheckpointConfig(**ckpt_cfg),
        inference=InferenceConfig(**inf_cfg)
    )