CSC3094 README/use case:

Training DreamerV3 for the first time:
Files needed; traindv3.py, serial_entry_mbrl.py

Lines to edit in traindv3.py;
19-name of run, 24-environment name, 18-73 for config, 105-113 for mlflow loggings,
116-max_env_step in serial_pipeline_dreamer call

Lines to edit in serial_entry_mbrl.py;
44-save location for replay gifs

To train DreamerV3 without a checkpoint file, simple alter above^ to requirements (notes about state size included in traindv3.py)
and then run traindv3.py


Training DreamerV3 from a checkpoint:
Files needed; traindv3.py, serial_entry_mbrl.py, retraindv3.py

Same lines to edit in both above files^, particular attention to altering environment name if you want to continue training on a different env.

Lines to edit in retraindv3.py
18-path to checkpoint file (pth.tar), 28-36 for mlflow loggings

To retrain DreamerV3, alter above^ and run retraindv3.py


Functionality:

Both training processes function the same; passing the configurations (minigrid_dreamer_config and minigrid_create_config to serial_pipeline_dreamer, which uses them to create instances of the world model,
combined policy, collector environment and evaluator environment. The evaluator environment has enable_save_replay set to a chosen path, which means that every evaluation cycle saves a gif replay of the
agent interacting with the environment. In addition, mlflow is used during the same evaluation section of serial_entry_dreamer to log the average reward for the evaluation cycle, allowing the mlflow metrics
to be directly compared against the gif replays.
If the evaluation cycle has hit a new best reward, then this state is saved as (model_name)./ckpt/ckpt_best.pth.tar, which can be used for loading the model for retraining.

To retrain, use the absolute file path of the .pth.tar checkpoint file and select the environment to be used (currently must be done in traindv3.py not retraindv3.py). This will initiate training of the DreamerV3
algorithm from the state in the checkpoint file.
