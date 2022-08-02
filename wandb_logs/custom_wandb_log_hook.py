import os
import pathlib
from contextlib import suppress
from datetime import datetime

import wandb
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import hooks
from colossalai.trainer.hooks._metric_hook import ThroughputMetric

# helper function
# def validate_input(param_to_validate):
#     try:
#         return list_to_append.append(param)
#     except AttributeError as e:
#         print('⚠️ WandBHook missing wandb tag: ', e)
#         return ''

class WandBHook(hooks.BaseHook):
    """
    Kastan custom WandB logger.
    See colossal AI docs for more hooks similar to `def after_train_iter()`
    """
    
    def __init__(self, args, gpc, priority=10):
        self.priority = priority
        self._logger = get_dist_logger()
        
        colossal_config_filepath = pathlib.Path(args.config)
        experiment_name = colossal_config_filepath.stem # just stem filename, no .filetype
        datetime_str = str(os.environ['EXPERIMENT_START_TIME']) # set from sbatch launcher
        
        # Suppresses errors when tags are missing! Brilliant! Must do line-by-line... 😑
        wandbtags: list[str] = []
        with suppress(AttributeError): wandbtags.append(datetime_str)
        with suppress(AttributeError): wandbtags.append(f"SLURM={os.environ['WANDB_SLURM_ID']}")
        with suppress(AttributeError): wandbtags.append(f"WORLD_SIZE={os.environ['WANDB_WORLD_SIZE']}")
        with suppress(AttributeError): wandbtags.append(f"TP={gpc.config.TENSOR_PARALLEL_SIZE}")
        with suppress(AttributeError): wandbtags.append(f"PP={gpc.config.PIPELINE_SIZE}")
        with suppress(AttributeError): wandbtags.append(f"NUM_EPOCHS={gpc.config.NUM_EPOCHS}")
        with suppress(AttributeError): wandbtags.append(f"BATCH_SIZE{gpc.config.BATCH_SIZE}")
        with suppress(AttributeError): wandbtags.append(f"MICRO_BATCH_SIZE={gpc.config.MICRO_BATCH_SIZE}")
        with suppress(AttributeError): wandbtags.append(f"NUM_MICRO_BATCHES={gpc.config.NUM_MICRO_BATCHES}")
        if len(wandbtags) == 0:
            wandbtags = ['no_tags']
    
        wandb.init(
            entity="kastan",
            project="LLM-Distributed-Quantization",
            config=gpc.config,
            name=experiment_name,
            group=datetime_str,
            tags=wandbtags,
        )

        try:
            ## Save whole model config file.
            wandb.save(args.config)  
        except Exception as e:
            print("WandBHook: Error saving colossalai config file:", e)
        
        try:
            # keep this after wandb.init()
            wandb.config.data_dir = os.environ['DATA']
            wandb.config.conda_env_name = os.environ['CONDA_ENV_NAME']
            wandb.config.num_gpus_per_node = os.environ['NUM_GPUS_PER_NODE']
            wandb.config.total_gpus = os.environ['WANDB_WORLD_SIZE']
        except KeyError:
            print("⚠️ WARNING: DATA environment variable not set. ⚠️")
            

    def after_train_iter(self, trainer, logits, label, loss):
        # copied from here https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/trainer/hooks/_log_hook.py#L39
        trainer.states['step_metrics'] = {}
        for metric_name, metric_calculator in trainer.states['metrics']['train'].items():
            if isinstance(metric_calculator, ThroughputMetric):
                trainer.states['step_metrics'][metric_name.lower()] = metric_calculator.get_last_step_info()
            else:
                trainer.states['step_metrics'][metric_name.lower()] = metric_calculator.get_last_step_value()


        # reformat throughput metrics (string --> 2 floats)
        metrics = trainer.states['step_metrics']
        # return a list of all floats in a string
        import re
        Tflops = None          # for when only 1 is returned
        samples_per_sec = None # for when only 1 is returned
        try:
            throughput_string = metrics['throughput']
            samples_per_sec, Tflops = re.findall("\d+\.\d+", throughput_string)

            del metrics['throughput']
            del metrics['lr']
        except ValueError as e:
            # probably Tflops is not available (in vanilla pytorch). Only collect sample_per_sec.
            samples_per_sec = re.findall("\d+\.\d+", throughput_string)[0] # only first value
            del metrics['throughput']
            del metrics['lr']
            print("ValueError: throughput not available", e)
        except Exception as e:
            # expecting the occational ValueError: not enough values to unpack (expected 2, got 1)
            print("Error when collecting samples_per_sec, Tflops...", e, "full metrics:", metrics)

        if samples_per_sec:
            metrics['samples_per_sec'] = float(samples_per_sec)
        if Tflops:
            metrics['Tflops'] = float(Tflops)
        # ^^ done cleaning

        # LOG TO WANDB
        wandb.log({ "loss": loss})
        wandb.log({ "per_step_metrics": metrics })

    def before_train(self, trainer):
        self._logger.info('training starts')

    def after_train(self, trainer):
        self._logger.info('training finished')
        wandb.run.summary["after_train"] = "Goodbye, we finished training"
