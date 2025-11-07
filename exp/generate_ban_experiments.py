#!/usr/bin/env python3
"""Generate BAN experiment configs and SLURM run scripts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from textwrap import dedent

BASE_TRAINING_PARAMS = {
    "seed": 1337,
    "subset_percent": 1.0,
    "augmentation": "yes",
    "batch_size": 128,
    "num_workers": 4,
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "nesterov": True,
    "lr_gamma": 0.2,
    "densenet_drop_rate": 0.0,
    "densenet_compression": 2,
    "label_smoothing": 0.0,
    "save_best_only": True,
    "save_summary_steps": 100,
}

DEFAULT_LR_MILESTONE_RATIOS = (0.3, 0.6, 0.8)
DEFAULT_WANDB_PROJECT = "BAN"
GENERATION_ORDER = ("vanilla", "gen1", "gen2", "gen3")

MODEL_SPECS = {
    "densenet80_80_cifar100": {
        "dataset": "cifar100",
        "model_version": "densenet_bc80_80",
        "distill_model_version": "densenet_bc80_80_distill",
        "teacher": "densenet_bc80_80",
    },
    "densenet80_120_cifar100": {
        "dataset": "cifar100",
        "model_version": "densenet_bc80_120",
        "distill_model_version": "densenet_bc80_120_distill",
        "teacher": "densenet_bc80_120",
    },
    "densenet90_60_cifar100": {
        "dataset": "cifar100",
        "model_version": "densenet_bc90_60",
        "distill_model_version": "densenet_bc90_60_distill",
        "teacher": "densenet_bc90_60",
    },
    "densenet112_33_cifar100": {
        "dataset": "cifar100",
        "model_version": "densenet_bc112",
        "distill_model_version": "densenet_bc112_distill",
        "teacher": "densenet_bc112",
    },
}

EXPERIMENT_RECIPES = {
    "ban": {
        "num_epochs": 200,
        "lr_milestone_ratios": DEFAULT_LR_MILESTONE_RATIOS,
        "training_overrides": {},
        "model_root_strategy": "legacy",
        "wandb_project": DEFAULT_WANDB_PROJECT,
    },
    "ban-e300": {
        "num_epochs": 300,
        "lr_milestone_ratios": DEFAULT_LR_MILESTONE_RATIOS,
        "training_overrides": {},
        "model_root_strategy": "scoped",
        "wandb_project": DEFAULT_WANDB_PROJECT,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BAN configs and run scripts."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENT_RECIPES.keys()),
        help="Experiment names to generate.",
    )
    parser.add_argument(
        "--arches",
        nargs="+",
        default=list(MODEL_SPECS.keys()),
        help="Architectures to include.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute targets without writing files.",
    )
    return parser.parse_args()


def prepare_recipe(raw_recipe: dict) -> dict:
    recipe = dict(raw_recipe)
    lr_milestone_ratios = recipe.pop("lr_milestone_ratios", DEFAULT_LR_MILESTONE_RATIOS)
    wandb_project = recipe.pop("wandb_project", DEFAULT_WANDB_PROJECT)
    model_root_strategy = recipe.pop("model_root_strategy", "scoped")
    kd_alpha = recipe.pop("kd_alpha", 1.0)
    kd_temperature = recipe.pop("kd_temperature", 1.0)

    training_overrides = dict(recipe.pop("training_overrides", {}))

    # Allow any BASE_TRAINING_PARAMS specified directly in the recipe to override defaults.
    for key in list(recipe.keys()):
        if key in BASE_TRAINING_PARAMS:
            training_overrides[key] = recipe.pop(key)

    if "num_epochs" not in recipe and "num_epochs" not in training_overrides:
        raise ValueError("Each recipe must define 'num_epochs'.")

    num_epochs = int(training_overrides.get("num_epochs", recipe.pop("num_epochs")))
    lr_milestones = [
        max(1, int(round(ratio * num_epochs)))
        for ratio in lr_milestone_ratios
    ]

    config_defaults = dict(BASE_TRAINING_PARAMS)
    config_defaults.update(training_overrides)
    config_defaults["num_epochs"] = num_epochs
    config_defaults["lr_milestones"] = lr_milestones

    prepared = dict(recipe)
    prepared.update({
        "lr_milestone_ratios": lr_milestone_ratios,
        "lr_milestones": lr_milestones,
        "num_epochs": num_epochs,
        "config_defaults": config_defaults,
        "wandb_project": wandb_project,
        "model_root_strategy": model_root_strategy,
        "kd_alpha": kd_alpha,
        "kd_temperature": kd_temperature,
    })
    return prepared


def resolve_model_stage_dir(
    models_root: Path, exp_name: str, stage_slug: str, strategy: str
) -> Path:
    if strategy == "legacy":
        return models_root / stage_slug
    return models_root / exp_name / stage_slug


def derive_arch_tag(arch_name: str, dataset: str) -> str:
    suffix = f"_{dataset}"
    if arch_name.endswith(suffix):
        return arch_name[: -len(suffix)]
    return arch_name


def build_stage_config(
    model_spec: dict,
    recipe: dict,
    stage_slug: str,
    arch_config_name: str,
) -> dict:
    config = dict(recipe["config_defaults"])
    config.update(
        {
            "dataset": model_spec["dataset"],
            "arch": arch_config_name,
            "model_version": (
                model_spec["model_version"]
                if stage_slug == "vanilla"
                else model_spec["distill_model_version"]
            ),
        }
    )
    if stage_slug == "vanilla":
        config["teacher"] = "none"
        config["use_kd"] = False
        config["alpha"] = 0.0
        config["temperature"] = 1.0
    else:
        config["teacher"] = model_spec["teacher"]
        config["use_kd"] = True
        config["alpha"] = recipe["kd_alpha"]
        config["temperature"] = recipe["kd_temperature"]
    return config


def build_stage_contexts(
    exp_name: str,
    recipe: dict,
    model_name: str,
    model_spec: dict,
    config_base: Path,
    models_root: Path,
) -> list[dict]:
    contexts: list[dict] = []
    prev_context: dict | None = None
    arch_tag = derive_arch_tag(model_name, model_spec["dataset"])
    for stage_slug in GENERATION_ORDER:
        arch_config_name = f"{stage_slug}_{model_name}"
        run_name = f"{exp_name}_{stage_slug}_{model_name}"
        config_dir = config_base / arch_config_name
        config_path = config_dir / f"{arch_config_name}.json"
        model_stage_dir = resolve_model_stage_dir(
            models_root, exp_name, stage_slug, recipe["model_root_strategy"]
        )
        model_run_dir = model_stage_dir / run_name
        best_checkpoint = model_run_dir / "best.pth.tar"
        teacher_ckpt = prev_context["best_checkpoint"] if prev_context else None
        if stage_slug != "vanilla" and teacher_ckpt is None:
            raise ValueError(
                f"Cannot determine teacher checkpoint for '{arch_config_name}'. "
                "Ensure the previous stage is defined."
            )
        config_payload = build_stage_config(
            model_spec, recipe, stage_slug, arch_config_name
        )
        wandb_tags = [
            exp_name,
            stage_slug,
            model_spec["dataset"],
            arch_tag,
            f"epoch{recipe['num_epochs']}",
        ]
        if stage_slug != "vanilla":
            wandb_tags.append("kd")
        context = {
            "stage_slug": stage_slug,
            "arch_name": model_name,
            "arch_config_name": arch_config_name,
            "arch_tag": arch_tag,
            "dataset": model_spec["dataset"],
            "run_name": run_name,
            "config_path": config_path,
            "config": config_payload,
            "model_stage_dir": model_stage_dir,
            "model_run_dir": model_run_dir,
            "best_checkpoint": best_checkpoint,
            "teacher_checkpoint": teacher_ckpt,
            "wandb_tags": wandb_tags,
        }
        contexts.append(context)
        prev_context = context
    return contexts


def write_json(path: Path, payload: dict, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)
        handle.write("\n")


def write_script(path: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.chmod(path, 0o755)


def render_stage_block(ctx: dict, wandb_project: str) -> str:
    tags = " ".join(ctx["wandb_tags"])
    teacher_cli = ""
    if ctx["teacher_checkpoint"] is not None:
        teacher_cli = (
            f'  --teacher_checkpoint "{ctx["teacher_checkpoint"].as_posix()}" \\\n'
        )
    block = f"""
echo ""
echo ">>> [{ctx['stage_slug'].upper()}] {ctx['arch_name']} ({ctx['dataset']})"
RUN_NAME="{ctx['run_name']}"
CONFIG_PATH="{ctx['config_path'].as_posix()}"
MODEL_STAGE_DIR="{ctx['model_stage_dir'].as_posix()}"
MODEL_DIR="{ctx['model_run_dir'].as_posix()}"
mkdir -p "{ctx['model_stage_dir'].as_posix()}"
mkdir -p "{ctx['model_run_dir'].as_posix()}"

export WANDB_RUN_NAME="${{RUN_NAME}}"

srun python -u train.py \\
  --model_dir="{ctx['model_run_dir'].as_posix()}" \\
  --config="{ctx['config_path'].as_posix()}" \\
  --enable_wandb true \\
  --wandb_mode online \\
  --project {wandb_project} \\
  --experiment "${{RUN_NAME}}" \\
{teacher_cli}  --seed 42 \\
  --wandb_tags {tags}
"""
    return dedent(block).strip()


def render_run_script(
    exp_name: str,
    arch_name: str,
    recipe: dict,
    stage_contexts: list[dict],
    repo_root: Path,
    run_out_dir: Path,
) -> str:
    job_name = f"{exp_name}-{arch_name}".replace("_", "-")
    # output_path = (run_out_dir / f"{arch_name}.out").as_posix()
    output_path = (run_out_dir / f"{arch_name}.out").as_posix()
    stage_blocks = "\n\n".join(
        render_stage_block(ctx, recipe["wandb_project"]) for ctx in stage_contexts
    )
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output={output_path}

set -euo pipefail

module purge
source ~/.bashrc
conda activate ban

cd {repo_root.as_posix()}

LOCAL_TMP=/tmp/${{USER}}_${{SLURM_JOB_ID}}
mkdir -p "${{LOCAL_TMP}}"
export TMPDIR="${{LOCAL_TMP}}"
trap 'rm -rf "${{LOCAL_TMP}}"' EXIT

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 29500-65000 -n 1)

export WANDB_PROJECT={recipe["wandb_project"]}
export WANDB_MODE=online

if [[ $SLURM_PROCID -eq 0 ]]; then
  wandb login 01126ae90da25bae0d86704140ac978cb9fd9c73
fi

mkdir -p "{run_out_dir.as_posix()}"

srun hostname -s | sort
srun bash -c 'echo "$SLURM_PROCID $(hostname -s): $CUDA_VISIBLE_DEVICES"'

{stage_blocks}
"""
    return dedent(script)


def render_run_all(script_paths: list[Path]) -> str:
    lines = ["#!/bin/bash", ""]
    for path in script_paths:
        lines.append(f"sbatch {path.name}")
    lines.append("")
    return "\n".join(lines)


def generate_experiment(
    exp_name: str,
    recipe: dict,
    arch_names: list[str],
    repo_root: Path,
    config_root: Path,
    run_root: Path,
    models_root: Path,
    dry_run: bool,
) -> tuple[int, int]:
    config_base = config_root / exp_name
    run_dir = run_root / exp_name
    run_out_dir = run_dir / "out"
    if not dry_run:
        run_out_dir.mkdir(parents=True, exist_ok=True)
    config_count = 0
    run_scripts: list[Path] = []
    for arch_name in arch_names:
        model_spec = MODEL_SPECS[arch_name]
        stage_contexts = build_stage_contexts(
            exp_name, recipe, arch_name, model_spec, config_base, models_root
        )
        for ctx in stage_contexts:
            write_json(ctx["config_path"], ctx["config"], dry_run)
            config_count += 1
        run_script_path = run_dir / f"train_{arch_name}.sh"
        script_body = render_run_script(
            exp_name, arch_name, recipe, stage_contexts, repo_root, run_out_dir
        )
        write_script(run_script_path, script_body, dry_run)
        run_scripts.append(run_script_path)
    run_all_path = run_dir / "run_all.sh"
    run_all_body = render_run_all(run_scripts)
    write_script(run_all_path, run_all_body, dry_run)
    return config_count, len(run_scripts) + 1  # include run_all


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    repo_root = project_root / "knowledge-distillation-pytorch"
    config_root = repo_root / "experiments"
    run_root = project_root / "exp"
    models_root = project_root / "models"

    exp_names = []
    for exp in args.experiments:
        if exp not in EXPERIMENT_RECIPES:
            raise SystemExit(f"Unknown experiment '{exp}'.")
        exp_names.append(exp)
    arch_names = []
    for arch in args.arches:
        if arch not in MODEL_SPECS:
            raise SystemExit(f"Unknown architecture '{arch}'.")
        arch_names.append(arch)

    for exp_name in exp_names:
        recipe = prepare_recipe(EXPERIMENT_RECIPES[exp_name])
        config_count, script_count = generate_experiment(
            exp_name,
            recipe,
            arch_names,
            repo_root,
            config_root,
            run_root,
            models_root,
            args.dry_run,
        )
        mode = "DRY" if args.dry_run else "WROTE"
        print(
            f"[{mode}] {exp_name}: {config_count} configs, {script_count} scripts (train + run_all)."
        )


if __name__ == "__main__":
    main()
