"""Main entrance for train/eval with/without KD on CIFAR-10"""

import argparse
import logging
import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm

import utils
import model.net as net
import model.data_loader as data_loader
import model.resnet as resnet
import model.wrn as wrn
import model.densenet as densenet
import model.resnext as resnext
import model.preresnet as preresnet
from evaluate import evaluate, evaluate_kd


def str2bool(v):
    """Convert common string inputs to booleans for argparse."""
    if isinstance(v, bool):
        return v
    val = v.lower()
    if val in ('yes', 'true', 't', 'y', '1'):
        return True
    if val in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory for the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--config', default=None,
                    help="Optional path to a params JSON file; overrides --model_dir")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--enable_wandb', type=str2bool, default=True,
                    help="Enable logging to Weights & Biases.")
parser.add_argument('--wandb_mode', type=str, default='offline',
                    choices=['online', 'offline', 'disabled'],
                    help="W&B mode: 'online' for internet connection, 'offline' for local logging, 'disabled' to turn off.")
parser.add_argument('--wandb_dir', type=str, default=None,
                    help="Directory to store W&B offline files (default: ~/.wandb).")
parser.add_argument('--project', default='ViT-s', type=str,
                    help="The name of the W&B project where you are sending the new run.")
parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                    help="Save model checkpoints as W&B Artifacts.")
parser.add_argument('--experiment', default=None, type=str,
                    help="Experiment name for both W&B run name and output directory subfolder.")
parser.add_argument('--seed', type=int, default=None,
                    help="Random seed for numpy, Python, and Torch. Defaults to value in params.json.")
parser.add_argument('--wandb_tags', nargs='*', default=None,
                    help="Optional list of tags to attach to the W&B run.")
parser.add_argument('--teacher_checkpoint', default=None,
                    help="Optional path to override the teacher checkpoint supplied in the JSON config.")
parser.add_argument('--dataset', type=str, default=None,
                    help="Dataset name when not supplying a params.json file.")
parser.add_argument('--arch', type=str, default=None,
                    help="Architecture identifier recorded in logs and W&B.")
parser.add_argument('--model_version', type=str, default=None,
                    help="Model variant to instantiate.")
parser.add_argument('--teacher', type=str, default=None,
                    help="Teacher model identifier (use 'none' for vanilla training).")
parser.add_argument('--subset_percent', type=float, default=None,
                    help="Fraction of training data to use (0-1).")
parser.add_argument('--augmentation', type=str, default=None,
                    help="Enable ('yes') or disable ('no') data augmentation.")
parser.add_argument('--batch_size', type=int, default=None,
                    help="Batch size for loaders.")
parser.add_argument('--num_workers', type=int, default=None,
                    help="Number of worker processes for DataLoader.")
parser.add_argument('--num_epochs', type=int, default=None,
                    help="Number of epochs to train.")
parser.add_argument('--learning_rate', type=float, default=None,
                    help="Base learning rate.")
parser.add_argument('--momentum', type=float, default=None,
                    help="Momentum value for SGD.")
parser.add_argument('--weight_decay', type=float, default=None,
                    help="Weight decay for optimizer.")
parser.add_argument('--nesterov', type=str2bool, default=None,
                    help="Use Nesterov momentum.")
parser.add_argument('--lr_milestones', type=int, nargs='*', default=None,
                    help="Space-separated learning rate milestone epochs.")
parser.add_argument('--lr_gamma', type=float, default=None,
                    help="Factor to decay learning rate at each milestone.")
parser.add_argument('--densenet_drop_rate', type=float, default=None,
                    help="Drop rate for DenseNet variants.")
parser.add_argument('--densenet_compression', type=float, default=None,
                    help="Compression rate for DenseNet variants.")
parser.add_argument('--alpha', type=float, default=None,
                    help="KD loss weight.")
parser.add_argument('--temperature', type=float, default=None,
                    help="KD temperature.")
parser.add_argument('--use_kd', type=str2bool, default=None,
                    help="Enable knowledge distillation training mode.")
parser.add_argument('--label_smoothing', type=float, default=None,
                    help="Label smoothing factor.")
parser.add_argument('--save_best_only', type=str2bool, default=None,
                    help="Store only best checkpoint.")
parser.add_argument('--save_summary_steps', type=int, default=None,
                    help="Steps between training log summaries.")


CLI_PARAM_FIELDS = (
    "dataset",
    "arch",
    "model_version",
    "teacher",
    "subset_percent",
    "augmentation",
    "batch_size",
    "num_workers",
    "num_epochs",
    "learning_rate",
    "momentum",
    "weight_decay",
    "nesterov",
    "lr_milestones",
    "lr_gamma",
    "densenet_drop_rate",
    "densenet_compression",
    "alpha",
    "temperature",
    "use_kd",
    "label_smoothing",
    "save_best_only",
    "save_summary_steps",
)

CLI_PARAM_DEFAULTS = {
    "teacher": "none",
    "subset_percent": 1.0,
    "augmentation": "yes",
    "batch_size": 128,
    "num_workers": 4,
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "nesterov": True,
    "lr_milestones": [60, 120, 160],
    "lr_gamma": 0.2,
    "densenet_drop_rate": 0.0,
    "densenet_compression": 2,
    "alpha": 0.0,
    "temperature": 1.0,
    "use_kd": False,
    "label_smoothing": 0.0,
    "save_best_only": True,
    "save_summary_steps": 100,
}

CLI_REQUIRED_FIELDS = ("dataset", "arch", "model_version", "num_epochs")


def _normalize_lr_milestones(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if isinstance(value, str):
        parts = [p for p in value.replace(",", " ").split() if p]
        return [int(p) for p in parts]
    return [int(value)]


def build_params_from_cli_args(args):
    """Assemble a Params instance from CLI overrides when --config is omitted."""
    data = {}
    for field in CLI_PARAM_FIELDS:
        value = getattr(args, field, None)
        if value is None:
            value = CLI_PARAM_DEFAULTS.get(field)
        if field == "lr_milestones" and value is not None:
            value = _normalize_lr_milestones(value)
        if value is not None:
            data[field] = value
    missing = [field for field in CLI_REQUIRED_FIELDS if field not in data or data[field] is None]
    if missing:
        raise ValueError(
            "Missing required hyperparameters when --config is not provided: {}".format(
                ", ".join(missing)
            )
        )
    return utils.Params.from_dict(data)


def compose_epoch_snapshot(epoch_idx, train_metrics, val_metrics):
    """Merge train/val metrics along with epoch metadata for serialization."""
    snapshot = dict(val_metrics)
    snapshot["epoch"] = epoch_idx
    if train_metrics:
        for key in ("train_loss", "train_ce_loss", "train_kd_loss", "train_grad_norm"):
            if key in train_metrics:
                snapshot[key] = train_metrics[key]
    ce_val = snapshot.get("ce_loss", snapshot.get("loss"))
    if ce_val is not None:
        snapshot.setdefault("val_ce_loss", ce_val)
    kd_val = snapshot.get("kd_loss")
    if kd_val is None:
        kd_val = 0.0
    snapshot["val_kd_loss"] = kd_val
    return snapshot


def seed_everything(seed=42):
    """Ensure reproducible behavior across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_grad_norm(parameters):
    """Return the L2 norm of gradients for a collection of parameters."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def _select_metric(metrics, keys):
    """Pick the first available metric value from keys and cast to float."""
    if not metrics:
        return None
    for key in keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, (np.generic, torch.Tensor)):
                value = float(value)
            return float(value)
    return None


def log_metrics_summary(scope, last_metrics, best_metrics, last_epoch=None, best_epoch=None):
    """Log a concise summary for final and best validation metrics."""
    if not last_metrics and not best_metrics:
        logging.info("%s summary unavailable (no metrics captured).", scope)
        return

    last_acc = _select_metric(last_metrics, ('test_acc1', 'accuracy', 'top1'))
    best_acc = _select_metric(best_metrics, ('test_acc1', 'accuracy', 'top1'))

    last_err = _select_metric(last_metrics, ('test_err1', 'error'))
    if last_err is None and last_acc is not None:
        last_err = 100.0 - last_acc
    best_err = _select_metric(best_metrics, ('test_err1', 'error'))
    if best_err is None and best_acc is not None:
        best_err = 100.0 - best_acc

    summary_parts = []
    if last_epoch is not None:
        summary_parts.append(f"last_epoch={last_epoch}")
    if last_acc is not None:
        summary_parts.append(f"last_acc={last_acc:05.3f}%")
    if last_err is not None:
        summary_parts.append(f"last_err={last_err:05.3f}%")
    if best_epoch is not None:
        summary_parts.append(f"best_epoch={best_epoch}")
    if best_acc is not None:
        summary_parts.append(f"best_acc={best_acc:05.3f}%")
    if best_err is not None:
        summary_parts.append(f"best_err={best_err:05.3f}%")

    if not summary_parts:
        logging.info("%s summary unavailable (metrics missing expected keys).", scope)
        return

    logging.info("%s summary -> %s", scope, " ; ".join(summary_parts))


DENSENET_FACTORY_MAP = {
    "densenet_bc112": densenet.DenseNetBC112_33,
    "densenet_bc90_60": densenet.DenseNetBC90_60,
    "densenet_bc80_80": densenet.DenseNetBC80_80,
    "densenet_bc80_120": densenet.DenseNetBC80_120,
}


def build_densenet_variant(model_version, params, wrap_dataparallel=True):
    """Instantiate a DenseNet variant defined in DENSENET_FACTORY_MAP."""
    base_version = model_version[:-8] if model_version.endswith('_distill') else model_version
    factory = DENSENET_FACTORY_MAP.get(base_version)
    if factory is None:
        return None
    model = factory(
        num_classes=params.num_classes,
        dropRate=getattr(params, 'densenet_drop_rate', 0.0),
        compressionRate=getattr(params, 'densenet_compression', 2))
    if params.cuda:
        if wrap_dataparallel:
            model = nn.DataParallel(model).cuda()
        else:
            model = model.cuda()
    return model


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: 
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    ce_loss_avg = utils.RunningAverage()
    kd_loss_avg = utils.RunningAverage()
    grad_norm_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch = train_batch.cuda(non_blocking=True)
                labels_batch = labels_batch.cuda(non_blocking=True)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)
            ce_loss = loss
            kd_loss_value = 0.0

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            grad_norm = compute_grad_norm(model.parameters())
            optimizer.step()

            loss_value = loss.item()
            ce_loss_value = ce_loss.item()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from tensors, move to cpu, convert to numpy arrays
                output_batch_np = output_batch.detach().cpu().numpy()
                labels_batch_np = labels_batch.detach().cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch_np, labels_batch_np)
                                 for metric in metrics}
                if 'accuracy' in summary_batch:
                    summary_batch['accuracy'] *= 100.0
                summary_batch['train_loss'] = loss_value
                summary_batch['train_ce_loss'] = ce_loss_value
                summary_batch['train_kd_loss'] = kd_loss_value
                summary_batch['train_grad_norm'] = grad_norm
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss_value)
            ce_loss_avg.update(ce_loss_value)
            kd_loss_avg.update(kd_loss_value)
            grad_norm_avg.update(grad_norm)

            t.set_postfix(train_loss='{:05.3f}'.format(loss_avg()),
                          grad_norm='{:05.3f}'.format(grad_norm_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                       loss_fn, metrics, params, model_dir, restore_file=None,
                       board_logger=None, wandb_ckpt=False):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    best_metrics_snapshot = None
    best_epoch = None
    last_metrics_snapshot = None
    completed_epoch = 0

    scheduler = None
    if params.model_version == "resnet18":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    elif params.model_version == "cnn":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version in DENSENET_FACTORY_MAP:
        milestones = getattr(params, 'lr_milestones', [60, 120, 160])
        gamma = getattr(params, 'lr_gamma', 0.2)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics = train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)
        snapshot = compose_epoch_snapshot(epoch + 1, train_metrics, val_metrics)
        last_metrics_snapshot = snapshot
        completed_epoch = epoch + 1

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        if board_logger and board_logger.enabled and wandb_ckpt:
            last_path = os.path.join(model_dir, 'last.pth.tar')
            board_logger.log_checkpoint(
                last_path,
                name="epoch-{}-last".format(epoch + 1),
                aliases=["last", "epoch-{}".format(epoch + 1)],
                metadata={"epoch": epoch + 1, "best": False})
            if is_best:
                best_path = os.path.join(model_dir, 'best.pth.tar')
                board_logger.log_checkpoint(
                    best_path,
                    name="epoch-{}-best".format(epoch + 1),
                    aliases=["best", "epoch-{}".format(epoch + 1)],
                    metadata={"epoch": epoch + 1, "best": True})

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            best_metrics_snapshot = dict(snapshot)
            best_epoch = epoch + 1

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(best_metrics_snapshot, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(snapshot, last_json_path)

        if board_logger and board_logger.enabled:
            train_payload = {"train/{}".format(k): v for k, v in train_metrics.items()}
            train_payload["lr"] = optimizer.param_groups[0]['lr']
            val_payload = {"val/{}".format(k): v for k, v in val_metrics.items()}
            board_logger.log_metrics(train_payload, step=epoch + 1)
            board_logger.log_metrics(val_payload, step=epoch + 1)

        if scheduler is not None:
            scheduler.step()

    log_metrics_summary("Validation", last_metrics_snapshot, best_metrics_snapshot,
                        last_epoch=completed_epoch or None,
                        best_epoch=best_epoch)


# Defining train_kd & train_and_evaluate_kd functions
def train_kd(model, teacher_model, optimizer, loss_fn_kd, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn_kd: 
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()
    teacher_model.eval()

    # summary for current training loop and running averages of tracked metrics
    summ = []
    loss_avg = utils.RunningAverage()
    ce_loss_avg = utils.RunningAverage()
    kd_loss_avg = utils.RunningAverage()
    grad_norm_avg = utils.RunningAverage()

    alpha = getattr(params, 'alpha', 0.0)

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch = train_batch.cuda(non_blocking=True)
                labels_batch = labels_batch.cuda(non_blocking=True)

            # compute model output, fetch teacher output, and compute KD loss
            output_batch = model(train_batch)

            # get one batch output from teacher_outputs list

            with torch.no_grad():
                output_teacher_batch = teacher_model(train_batch)
            if params.cuda:
                output_teacher_batch = output_teacher_batch.cuda(non_blocking=True)

            loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)
            ce_loss = F.cross_entropy(output_batch, labels_batch)
            kd_loss_tensor = loss - ce_loss * (1.0 - alpha)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            grad_norm = compute_grad_norm(model.parameters())
            optimizer.step()

            loss_value = loss.item()
            ce_loss_value = ce_loss.item()
            kd_loss_value = kd_loss_tensor.detach().item() if torch.is_tensor(kd_loss_tensor) else float(kd_loss_tensor)

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch_np = output_batch.detach().cpu().numpy()
                labels_batch_np = labels_batch.detach().cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch_np, labels_batch_np)
                                 for metric in metrics}
                if 'accuracy' in summary_batch:
                    summary_batch['accuracy'] *= 100.0
                summary_batch['train_loss'] = loss_value
                summary_batch['train_ce_loss'] = ce_loss_value
                summary_batch['train_kd_loss'] = kd_loss_value
                summary_batch['train_grad_norm'] = grad_norm
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss_value)
            ce_loss_avg.update(ce_loss_value)
            kd_loss_avg.update(kd_loss_value)
            grad_norm_avg.update(grad_norm)

            t.set_postfix(train_loss='{:05.3f}'.format(loss_avg()),
                          grad_norm='{:05.3f}'.format(grad_norm_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean


def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                          loss_fn_kd, metrics, params, model_dir, restore_file=None,
                          board_logger=None, wandb_ckpt=False):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - file to restore (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    best_metrics_snapshot = None
    best_epoch = None
    last_metrics_snapshot = None
    completed_epoch = 0
    
    # Tensorboard logger setup
    # board_logger = utils.Board_Logger(os.path.join(model_dir, 'board_logs'))

    # learning rate schedulers for different models:
    scheduler = None
    if params.model_version == "resnet18_distill":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    elif params.model_version == "cnn_distill":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    elif params.model_version.endswith("_distill") and params.model_version[:-8] in DENSENET_FACTORY_MAP:
        milestones = getattr(params, 'lr_milestones', [60, 120, 160])
        gamma = getattr(params, 'lr_gamma', 0.2)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    for epoch in range(params.num_epochs):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics = train_kd(model, teacher_model, optimizer, loss_fn_kd, train_dataloader,
                                 metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate_kd(model, teacher_model, loss_fn_kd, val_dataloader, metrics, params)
        snapshot = compose_epoch_snapshot(epoch + 1, train_metrics, val_metrics)
        last_metrics_snapshot = snapshot
        completed_epoch = epoch + 1

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        if board_logger and board_logger.enabled and wandb_ckpt:
            last_path = os.path.join(model_dir, 'last.pth.tar')
            board_logger.log_checkpoint(
                last_path,
                name="epoch-{}-last".format(epoch + 1),
                aliases=["last", "epoch-{}".format(epoch + 1)],
                metadata={"epoch": epoch + 1, "best": False})
            if is_best:
                best_path = os.path.join(model_dir, 'best.pth.tar')
                board_logger.log_checkpoint(
                    best_path,
                    name="epoch-{}-best".format(epoch + 1),
                    aliases=["best", "epoch-{}".format(epoch + 1)],
                    metadata={"epoch": epoch + 1, "best": True})

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            best_metrics_snapshot = dict(snapshot)
            best_epoch = epoch + 1

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(best_metrics_snapshot, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(snapshot, last_json_path)

        if board_logger and board_logger.enabled:
            train_payload = {"train/{}".format(k): v for k, v in train_metrics.items()}
            train_payload["lr"] = optimizer.param_groups[0]['lr']
            val_payload = {"val/{}".format(k): v for k, v in val_metrics.items()}
            board_logger.log_metrics(train_payload, step=epoch + 1)
            board_logger.log_metrics(val_payload, step=epoch + 1)

        if scheduler is not None:
            scheduler.step()

    log_metrics_summary("Validation", last_metrics_snapshot, best_metrics_snapshot,
                        last_epoch=completed_epoch or None,
                        best_epoch=best_epoch)


        # #============ TensorBoard logging: uncomment below to turn in on ============#
        # # (1) Log the scalar values
        # info = {
        #     'val accuracy': val_acc
        # }

        # for tag, value in info.items():
        #     board_logger.scalar_summary(tag, value, epoch+1)

        # # (2) Log values and gradients of the parameters (histogram)
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     board_logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
        #     # board_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    default_model_dir = parser.get_default('model_dir')

    # Support either a directory containing params.json or a direct JSON file path.
    if args.config:
        json_path = args.config
        config_dir = os.path.dirname(json_path) or '.'
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = utils.Params(json_path)
    else:
        json_path = None
        config_dir = args.model_dir or parser.get_default('model_dir') or '.'
        params = build_params_from_cli_args(args)

    if args.teacher_checkpoint:
        params.teacher_checkpoint = args.teacher_checkpoint

    params.dataset = getattr(params, 'dataset', 'cifar10')
    dataset_info = data_loader.get_dataset_info(params.dataset)
    if not hasattr(params, 'num_classes'):
        params.num_classes = dataset_info['num_classes']
    if not hasattr(params, 'data_root'):
        params.data_root = dataset_info['root']

    seed_from_config = getattr(params, 'seed', 42)

    user_supplied_model_dir = args.model_dir and (args.model_dir != default_model_dir)
    output_root = args.model_dir if user_supplied_model_dir else config_dir
    if not output_root:
        output_root = config_dir

    if args.experiment:
        root_leaf = os.path.basename(os.path.normpath(output_root))
        if root_leaf == args.experiment:
            output_dir = output_root
        else:
            output_dir = os.path.join(output_root, args.experiment)
    else:
        output_dir = output_root
    os.makedirs(output_dir, exist_ok=True)
    args.model_dir = output_dir

    seed_value = args.seed if args.seed is not None else seed_from_config
    params.seed = seed_value

    wandb_logger = None
    run_tags = []
    if args.wandb_tags:
        for tag in args.wandb_tags:
            tag = str(tag).strip()
            if tag and tag not in run_tags:
                run_tags.append(tag)

    auto_tags = []
    dataset_tag = params.dataset
    if dataset_tag:
        auto_tags.append(dataset_tag)
    model_version_tag = params.model_version
    if model_version_tag:
        auto_tags.append(model_version_tag)
    arch_tag = getattr(params, 'arch', None)
    if arch_tag:
        auto_tags.append(arch_tag)

    batch_tag = f"bs{getattr(params, 'batch_size', 'na')}"
    lr_tag = f"lr{getattr(params, 'learning_rate', 'na')}"
    seed_tag = f"seed{seed_value}"
    epoch_tag = f"epochs{getattr(params, 'num_epochs', 'na')}"

    auto_tags.extend([batch_tag, lr_tag, seed_tag, epoch_tag])

    kd_enabled = getattr(params, 'use_kd', False) or ('distill' in params.model_version)
    if kd_enabled:
        auto_tags.append('kd')
        alpha = getattr(params, 'alpha', None)
        temperature = getattr(params, 'temperature', None)
        teacher = getattr(params, 'teacher', None)
        if alpha is not None:
            auto_tags.append(f"alpha{alpha}")
        if temperature is not None:
            auto_tags.append(f"T{temperature}")
        if teacher and teacher.lower() != 'none':
            auto_tags.append(f"teacher_{teacher}")
    else:
        auto_tags.append('vanilla')

    scheduler_tag = getattr(params, 'lr_scheduler', None)
    if not scheduler_tag:
        base_version = params.model_version.replace('_distill', '')
        if base_version in DENSENET_FACTORY_MAP:
            milestones = getattr(params, 'lr_milestones', [])
            scheduler_tag = "multistep-" + "-".join(str(m) for m in milestones) if milestones else "multistep"
        elif 'cnn' in params.model_version:
            scheduler_tag = "step-100"
        elif 'resnet18' in params.model_version:
            scheduler_tag = "step-150"
    if scheduler_tag:
        auto_tags.append(f"sched_{scheduler_tag}")

    def add_tag(tag):
        tag = str(tag).strip().replace(' ', '_')
        if tag and tag not in run_tags:
            run_tags.append(tag)

    for tag in auto_tags:
        add_tag(tag)
    if args.enable_wandb and args.wandb_mode != 'disabled':
        run_name = args.experiment or os.path.basename(os.path.normpath(args.model_dir))
        wandb_config = dict(params.dict)
        wandb_config.update({
            "restore_file": args.restore_file,
            "experiment": args.experiment,
            "wandb_mode": args.wandb_mode,
            "wandb_ckpt": args.wandb_ckpt,
            "wandb_tags": run_tags
        })
        wandb_logger = utils.Board_Logger(
            log_dir=args.wandb_dir,
            project=args.project,
            run_name=run_name,
            config=wandb_config,
            mode=args.wandb_mode,
            tags=run_tags)
    params.wandb_tags = run_tags

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    seed_everything(seed_value)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    if args.experiment:
        logging.info("Writing outputs to experiment directory: %s", args.model_dir)
    if wandb_logger and wandb_logger.enabled:
        logging.info("Weights & Biases logging enabled (project=%s, mode=%s)", args.project, args.wandb_mode)
        if run_tags:
            logging.info("W&B tags: %s", ", ".join(run_tags))
    elif not wandb_logger:
        logging.info("Weights & Biases logging disabled.")

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders, considering full-set vs. sub-set scenarios
    if params.subset_percent < 1.0:
        train_dl = data_loader.fetch_subset_dataloader('train', params)
    else:
        train_dl = data_loader.fetch_dataloader('train', params)

    dev_dl = data_loader.fetch_dataloader('dev', params)

    logging.info("- done.")

    """Based on the model_version, determine model/optimizer and KD training mode
       WideResNet and DenseNet were trained on multi-GPU; need to specify a dummy
       nn.DataParallel module to correctly load the model parameters
    """
    if "distill" in params.model_version:

        # train a 5-layer CNN or a 18-layer ResNet with knowledge distillation
        if params.model_version == "cnn_distill":
            model = net.Net(params).cuda() if params.cuda else net.Net(params)
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = net.metrics

        elif params.model_version == 'resnet18_distill':
            student_resnet = resnet.ResNet18(num_classes=params.num_classes)
            model = student_resnet.cuda() if params.cuda else student_resnet
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics definition in model files
            loss_fn_kd = net.loss_fn_kd
            metrics = resnet.metrics
        else:
            student_densenet = build_densenet_variant(params.model_version, params, wrap_dataparallel=True)
            if student_densenet is None:
                raise ValueError("Unsupported distillation model_version '{}'".format(params.model_version))
            model = student_densenet
            optimizer = optim.SGD(
                model.parameters(),
                lr=params.learning_rate,
                momentum=getattr(params, 'momentum', 0.9),
                weight_decay=getattr(params, 'weight_decay', 5e-4),
                nesterov=getattr(params, 'nesterov', False))
            loss_fn_kd = net.loss_fn_kd
            metrics = densenet.metrics

        """ 
            Specify the pre-trained teacher models for knowledge distillation
            Important note: wrn/densenet/resnext/preresnet were pre-trained models using multi-GPU,
            therefore need to call "nn.DaraParallel" to correctly load the model weights
            Trying to run on CPU will then trigger errors (too time-consuming anyway)!
        """
        teacher_checkpoint = getattr(params, 'teacher_checkpoint', None)

        if params.teacher == "resnet18":
            teacher_resnet = resnet.ResNet18(num_classes=params.num_classes)
            teacher_model = teacher_resnet
            if teacher_checkpoint is None:
                teacher_checkpoint = 'experiments/base_resnet18/best.pth.tar'
            teacher_model = teacher_model.cuda() if params.cuda else teacher_model

        elif params.teacher == "wrn":
            teacher_model = wrn.WideResNet(depth=28, num_classes=params.num_classes, widen_factor=10,
                                           dropRate=0.3)
            if teacher_checkpoint is None:
                teacher_checkpoint = 'experiments/base_wrn/best.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()

        elif params.teacher == "densenet":
            teacher_model = densenet.DenseNet(depth=100, growthRate=12, num_classes=params.num_classes)
            if teacher_checkpoint is None:
                teacher_checkpoint = 'experiments/base_densenet/best.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()
        elif params.teacher in DENSENET_FACTORY_MAP:
            drop_rate = getattr(params, 'densenet_drop_rate', 0.0)
            compression = getattr(params, 'densenet_compression', 2)
            teacher_model = DENSENET_FACTORY_MAP[params.teacher](
                num_classes=params.num_classes,
                dropRate=drop_rate,
                compressionRate=compression)
            teacher_model = nn.DataParallel(teacher_model).cuda()

        elif params.teacher == "resnext29":
            teacher_model = resnext.CifarResNeXt(cardinality=8, depth=29, num_classes=params.num_classes)
            if teacher_checkpoint is None:
                teacher_checkpoint = 'experiments/base_resnext29/best.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()

        elif params.teacher == "preresnet110":
            teacher_model = preresnet.PreResNet(depth=110, num_classes=params.num_classes)
            if teacher_checkpoint is None:
                teacher_checkpoint = 'experiments/base_preresnet110/best.pth.tar'
            teacher_model = nn.DataParallel(teacher_model).cuda()

        else:
            raise ValueError("Unsupported teacher '{}' for dataset {}".format(
                params.teacher, params.dataset))

        if teacher_checkpoint is None:
            raise ValueError(
                "No teacher_checkpoint provided for teacher '{}' on dataset '{}'. "
                "Add 'teacher_checkpoint' to params.json or supply a compatible checkpoint."
                .format(params.teacher, params.dataset))

        utils.load_checkpoint(teacher_checkpoint, teacher_model)

        # Train the model with KD
        logging.info("Experiment - model version: {}".format(params.model_version))
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        logging.info("First, loading the teacher model and computing its outputs...")
        train_and_evaluate_kd(model, teacher_model, train_dl, dev_dl, optimizer, loss_fn_kd,
                              metrics, params, args.model_dir, args.restore_file,
                              board_logger=wandb_logger, wandb_ckpt=args.wandb_ckpt)

    # non-KD mode: regular training of the baseline CNN or ResNet-18
    else:
        if params.model_version == "cnn":
            model = net.Net(params).cuda() if params.cuda else net.Net(params)
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
            # fetch loss function and metrics
            loss_fn = net.loss_fn
            metrics = net.metrics

        elif params.model_version == "resnet18":
            baseline_resnet = resnet.ResNet18(num_classes=params.num_classes)
            model = baseline_resnet.cuda() if params.cuda else baseline_resnet
            optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                  momentum=0.9, weight_decay=5e-4)
            # fetch loss function and metrics
            loss_fn = resnet.loss_fn
            metrics = resnet.metrics
        elif params.model_version in DENSENET_FACTORY_MAP:
            base_densenet = build_densenet_variant(params.model_version, params, wrap_dataparallel=True)
            if base_densenet is None:
                raise ValueError("Unsupported model_version '{}'".format(params.model_version))
            model = base_densenet
            optimizer = optim.SGD(model.parameters(),
                                  lr=params.learning_rate,
                                  momentum=getattr(params, 'momentum', 0.9),
                                  weight_decay=getattr(params, 'weight_decay', 5e-4),
                                  nesterov=getattr(params, 'nesterov', False))
            loss_fn = densenet.loss_fn
            metrics = densenet.metrics

        # elif params.model_version == "wrn":
        #     model = wrn.wrn(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
        #     model = model.cuda() if params.cuda else model
        #     optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
        #                           momentum=0.9, weight_decay=5e-4)
        #     # fetch loss function and metrics
        #     loss_fn = wrn.loss_fn
        #     metrics = wrn.metrics

        # Train the model
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, params,
                           args.model_dir, args.restore_file,
                           board_logger=wandb_logger, wandb_ckpt=args.wandb_ckpt)

    if wandb_logger:
        wandb_logger.finish()
