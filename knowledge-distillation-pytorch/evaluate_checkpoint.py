#!/usr/bin/env python3
"""Utility script to evaluate a saved checkpoint on CIFAR datasets."""

import argparse
import json
import logging
import os
from typing import Tuple

import torch
import torch.nn as nn

import utils
import model.net as net
import model.resnet as resnet
import model.densenet as densenet
import model.data_loader as data_loader
from evaluate import evaluate


DENSENET_FACTORY_MAP = {
    "densenet_bc112": densenet.DenseNetBC112_33,
    "densenet_bc90_60": densenet.DenseNetBC90_60,
    "densenet_bc80_80": densenet.DenseNetBC80_80,
    "densenet_bc80_120": densenet.DenseNetBC80_120,
}


def _unwrap_version(version: str) -> str:
    """Return the base model name without distill suffix."""
    if version.endswith("_distill"):
        return version[:-8]
    return version


def build_model(params, use_dataparallel=True):
    """Instantiate the architecture defined in params."""
    version = getattr(params, "model_version", None)
    if not version:
        raise ValueError("params.model_version is required to build the model.")

    base_version = _unwrap_version(version)

    if base_version == "cnn":
        model = net.Net(params)
        loss_fn = net.loss_fn
        metrics = net.metrics
    elif base_version == "resnet18":
        model = resnet.ResNet18(num_classes=params.num_classes)
        loss_fn = resnet.loss_fn
        metrics = resnet.metrics
    elif base_version in DENSENET_FACTORY_MAP:
        model = DENSENET_FACTORY_MAP[base_version](
            num_classes=params.num_classes,
            dropRate=getattr(params, "densenet_drop_rate", 0.0),
            compressionRate=getattr(params, "densenet_compression", 2),
        )
        loss_fn = densenet.loss_fn
        metrics = densenet.metrics
        if params.cuda and use_dataparallel:
            model = nn.DataParallel(model).cuda()
        elif params.cuda:
            model = model.cuda()
        return model, loss_fn, metrics
    else:
        raise ValueError(f"Unsupported model_version '{version}'")

    if params.cuda:
        model = model.cuda()
    return model, loss_fn, metrics


def configure_logger(log_file: str = None):
    """Configure root logger."""
    handlers = []
    formatter = logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("--config", required=True, help="Path to params.json.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth.tar checkpoint.")
    parser.add_argument("--split", default="dev", choices=["train", "dev"],
                        help="Dataset split to evaluate on.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size defined in config.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override number of data loader workers.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Force evaluation on CPU.")
    parser.add_argument("--no_dataparallel", action="store_true",
                        help="Do not wrap models with nn.DataParallel (useful for CPU eval).")
    parser.add_argument("--output_json", default=None,
                        help="Optional path to store metrics JSON.")
    parser.add_argument("--log_file", default=None,
                        help="Optional path for a log file.")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_logger(args.log_file)

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    params = utils.Params(args.config)
    params.dataset = getattr(params, "dataset", "cifar10")
    dataset_info = data_loader.get_dataset_info(params.dataset)

    if not hasattr(params, "num_classes"):
        params.num_classes = dataset_info["num_classes"]
    if not hasattr(params, "data_root"):
        params.data_root = dataset_info["root"]

    if args.batch_size is not None:
        params.batch_size = args.batch_size
    if args.num_workers is not None:
        params.num_workers = args.num_workers

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    params.cuda = use_cuda

    logging.info("Loading model version '%s' for dataset '%s'", params.model_version, params.dataset)
    model, loss_fn, metrics = build_model(params, use_dataparallel=not args.no_dataparallel)

    logging.info("Loading checkpoint from %s", args.checkpoint)
    utils.load_checkpoint(args.checkpoint, model)

    logging.info("Preparing %s dataloader (batch_size=%s, workers=%s)",
                 args.split, params.batch_size, params.num_workers)
    dataloader = data_loader.fetch_dataloader(args.split, params)

    logging.info("Starting evaluation...")
    eval_metrics = evaluate(model, loss_fn, dataloader, metrics, params)
    formatted_metrics = json.dumps(eval_metrics, indent=2)
    logging.info("Evaluation metrics:\n%s", formatted_metrics)
    print(formatted_metrics)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        utils.save_dict_to_json(eval_metrics, args.output_json)
        logging.info("Saved metrics to %s", args.output_json)


if __name__ == "__main__":
    main()
