"""Helper utilities for experiment management and logging."""
import json
import logging
import os
import shutil

import numpy as np
import torch
import wandb


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("Checkpoint file does not exist: {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


class Board_Logger(object):
    """Weights & Biases logging utility."""

    def __init__(self, log_dir=None, project=None, run_name=None, config=None,
                 mode=None, tags=None, **kwargs):
        """Initialise a wandb run, respecting existing context when present."""
        self.run = wandb.run
        self.owns_run = False
        self.tags = tags

        if self.run is None:
            init_kwargs = {}
            if project:
                init_kwargs["project"] = project
            if run_name:
                init_kwargs["name"] = run_name
            if log_dir:
                init_kwargs["dir"] = log_dir
            if config:
                init_kwargs["config"] = config
            if mode:
                init_kwargs["mode"] = mode
            if tags:
                init_kwargs["tags"] = tags
            init_kwargs.update(kwargs)
            self.run = wandb.init(**init_kwargs)
            self.owns_run = True
        elif config:
            wandb.config.update(config, allow_val_change=True)

    @property
    def enabled(self):
        return self.run is not None

    def log_metrics(self, metrics, step=None):
        """Log a dictionary of metrics to Weights & Biases."""
        if not self.enabled:
            return
        wandb.log(metrics, step=step)

    def scalar_summary(self, tag, value, step=None):
        """Log a scalar value."""
        self.log_metrics({tag: value}, step=step)

    def image_summary(self, tag, images, step=None):
        """Log a list of images."""
        if not self.enabled:
            return
        log_payload = {f"{tag}/{i}": wandb.Image(img) for i, img in enumerate(images)}
        wandb.log(log_payload, step=step)

    def histo_summary(self, tag, values, step=None, bins=1000):
        """Log a histogram of the tensor of values."""
        if not self.enabled:
            return

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        hist_data = wandb.Histogram(np_histogram=(counts, bin_edges))
        wandb.log({tag: hist_data}, step=step)

    def log_checkpoint(self, path, name, aliases=None, metadata=None):
        """Persist a model checkpoint as a W&B Artifact."""
        if not self.enabled or not os.path.exists(path):
            return

        run_label = self.run.name or self.run.id or "run"
        safe_run = run_label.replace(" ", "_")
        safe_name = name.replace(" ", "_")
        artifact_name = f"{safe_run}-{safe_name}"
        artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata or {})
        artifact.add_file(path)
        self.run.log_artifact(artifact, aliases=aliases or [])

    def finish(self):
        """Finalize the W&B run if this logger created it."""
        if self.owns_run and self.enabled:
            wandb.finish()
            self.run = None
