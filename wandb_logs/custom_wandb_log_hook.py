import os
import pathlib
from datetime import datetime

import wandb
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import hooks
from colossalai.trainer.hooks._metric_hook import ThroughputMetric


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

        datetime_str = datetime.now().strftime("%h-%d__%H:%M") # group by
        wandbtags = 'my_first_tag' # tags 
        
        wandbtags : list[str] = [f'TP={gpc.config.TENSOR_PARALLEL_SIZE}', f'PP={gpc.config.PIPELINE_SIZE}', f'BATCH_SIZE{gpc.config.BATCH_SIZE}', f'NUM_EPOCHS={gpc.config.NUM_EPOCHS}', f'MICRO_BATCH_SIZE={gpc.config.MICRO_BATCH_SIZE}', f'NUM_MICRO_BATCHES={gpc.config.NUM_MICRO_BATCHES}']
        
        print("Our GPC config!! See what to steal here.")
        
        wandb.init(entity="kastan", project="LLM-Distributed-Quantization", config=gpc.config, name=experiment_name, group=datetime_str, tags=wandbtags)
        
        try:
            wandb.config.data_dir = os.environ['DATA']
        except KeyError:
            print("⚠️ WARNING: DATA environment variable not set. ⚠️")
            
        # wandb.run.summary["hello_message"] = "Hi, we got started"

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
