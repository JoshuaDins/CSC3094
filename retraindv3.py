import os
import gym
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import torch
from ding.config import compile_config
from ding.entry import serial_pipeline_dreamer
from ding.model import DREAMERVAC
from ding.policy import DREAMERPolicy
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from traindv3 import minigrid_dreamer_config, minigrid_create_config
import mlflow

mlflow.start_run
minigrid_dreamer_config['env']['id']= "MiniGrid-DoorKey-8x8-v0"
#C:\Users\joshd\University\CSC3094\DI-engine\dv3_grid_empty3\ckpt\ckpt_best.pth.tar
#model_path = 'C:/Users/joshd/University/CSC3094/DI-engine/dv3_retrain/ckpt/ckpt_best.pth.tar'
model_path='C:/Users/joshd/University/CSC3094/DI-engine/dv3_grid_empty3/ckpt/ckpt_best.pth.tar'
state_dict= torch.load(model_path, map_location="cpu")

cfg = compile_config(minigrid_dreamer_config, seed=0, env=None, auto=True, create_cfg=minigrid_create_config, save_cfg=True)

model= DREAMERVAC(**cfg.policy.model)
model.load_state_dict(state_dict['model'])
policy= DREAMERPolicy(cfg.policy, model=model)

if __name__ =="__main__":
    mlflow.log_param('env_id', 'MiniGrid-DoorKey-8x8-v0')
    mlflow.log_param('max_step', 100000)
    mlflow.log_param('learning_rate', 3e-5)
    mlflow.log_param('discount_factor', 0.997)
    mlflow.log_param('hidden_layer_sizes', [256, 128, 64, 64])
    mlflow.log_param('batch_size', 16)
    mlflow.log_param('eval_freq', 250)
    mlflow.log_param('exp_name', 'dv3_retrain.DK')
    mlflow.log_param('random_seed', 0)

    serial_pipeline_dreamer((minigrid_dreamer_config, minigrid_create_config), seed=0, env_setting=None, model=model)

    mlflow.end_run
