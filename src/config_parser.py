import os
import sys
import yaml
from dataclasses import dataclass
from typing import List

# Setup path access across the entire project
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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

@dataclass
class InferenceConfig:
    image_path: str
    model_path: str
    output_dir: str
    class_names: List[str]

@dataclass
class Config:
    dataset: DatasetConfig
    training: TrainingConfig
    checkpoint: CheckpointConfig
    inference: InferenceConfig

def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)
        
    return Config(
        dataset=DatasetConfig(**raw_cfg['dataset']),
        training=TrainingConfig(**raw_cfg['training']),
        checkpoint=CheckpointConfig(**raw_cfg['checkpoint']),
        inference=InferenceConfig(**raw_cfg['inference'])
    )