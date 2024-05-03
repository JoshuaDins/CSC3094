from easydict import EasyDict
from ding.entry import serial_pipeline_dreamer
from dizoo.minigrid.envs import MiniGridEnv
import numpy as np
import mlflow
import os
import torch
import numpy as np
import mlflow.pytorch

#start MLflow run
mlflow.start_run()

cuda = True
collector_env_num = 8
evaluator_env_num = 5

minigrid_dreamer_config = dict(                            #change env
    exp_name='DV3RDBD',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        env_id="MiniGrid-RedBlueDoors-8x8-v0",
        max_step=300,
        #stop_value=20,
        flat_obs=True,
        full_obs=True,
        onehot_obs=True,
        move_bonus=True,
    ),
    policy=dict(
        on_policy=False,
        multi_gpu=False,
        grad_clip=100,
        cuda=cuda,
        random_collect_size=2500,
        model=dict(
            action_shape=7,
            actor_dist='onehot',
        ),
        learn=dict(                     #alter for optimisation, was 0.95, 3e-5, 16, 64, 0.997
            lambda_=0.95,
            learning_rate=3e-5,
            batch_size=16,
            batch_length=64,
            imag_sample=True,
            discount=0.997,
            reward_EMA=True,

        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            action_size=7,
            collect_dyn_sample=True,
        ),
        eval=dict(evaluator=dict(eval_freq=1000)),
        other=dict(
            replay_buffer=dict(replay_buffer_size=200000, periodic_thruput_seconds=60),
        ),
    ),
    world_model=dict(
        pretrain=100,
        train_freq=2,
        cuda=cuda,
        model=dict(
            state_size=2688,                                        #normally for 8x8 is 1344, 7581 for 16x16, 5376 for obsmaze
            obs_type='vector',
            action_size=7,
            action_type='discrete',
            #encoder_hidden_size_list=[200, 200, 200, 50, 7],
            encoder_hidden_size_list=[256, 128, 64, 64],
            reward_size=1,
            batch_size=16,
        ),
    ),
)

minigrid_dreamer_config = EasyDict(minigrid_dreamer_config)

minigrid_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
        manager=dict(
            enable_save_replay=True,
        )
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='dreamer',
        import_names=['ding.policy.mbpolicy.dreamer'],
    ),
    replay_buffer=dict(type='sequence'),
    world_model=dict(
        type='dreamer',

        import_names=['ding.world_model.dreamer'],
    ),
)
minigrid_create_config = EasyDict(minigrid_create_config)


if __name__ == '__main__':
    #logging
    mlflow.log_param('env_id', 'MiniGrid-RedBlueDoors-8x8-v0')
    mlflow.log_param('max_step', 100000)
    mlflow.log_param('learning_rate', 3e-5)
    mlflow.log_param('discount_factor', 0.997)
    mlflow.log_param('hidden_layer_sizes', [200, 200, 200, 50, 7])
    mlflow.log_param('batch_size', 16)
    mlflow.log_param('eval_freq', 1000)
    mlflow.log_param('exp_name', 'DV3RDBD')
    mlflow.log_param('random_seed', 0)

    #train, reward logged in train loop
    policy= serial_pipeline_dreamer((minigrid_dreamer_config, minigrid_create_config), seed=0, max_env_step=1000000)
    torch.save(policy.model)

    torch.save(policy._model, 'policy_params.pth')
    model= policy._model
    print(model)
    total_params = sum(p.numel() for p in policy._model.parameters())
    print("Total params: {total_params}")

    # End MLflow run
    mlflow.end_run()
