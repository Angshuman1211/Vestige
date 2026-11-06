"""
Fine-tuning script for VESTIGE Phase 2 ablation.

Runs either the MLM baseline or DAM proposed fine-tuning run depending
on the config file passed. Both runs use identical hyperparameters —
only the data_collator differs.

Usage (from project root):
    python training/train.py training/config_mlm.yaml
    python training/train.py training/config_dam.yaml

Outputs (per run):
    training/checkpoints/<run_name>/          model checkpoints
    training/checkpoints/<run_name>/results.json  final eval metrics
"""
import os
# transformers auto-loads its TensorFlow backend, which is broken with the
# installed Keras 3 (see requirements.txt). Force the torch-only path before any
# transformers import, or the lazy loader fails with misleading import errors
# (e.g. "cannot import name 'GenerationMixin'").
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import json
import sys
import types
from pathlib import Path

import yaml

# triton is CUDA-only — unavailable on macOS. Both transformers and DNABERT-2
# check for it at import time. The stub must have __spec__ set or
# importlib.util.find_spec() raises ValueError.
import importlib.util, importlib.machinery
# triton stub removed: it broke transformers 4.57 import; DNABERT-2 falls back to torch attention.

import torch
from datasets import load_from_disk
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# DNABERT-2's remote code lists triton as a required import but falls back to
# PyTorch attention when it is absent. Drop triton from the dependency check so
# the model loads where triton is missing or incompatible (e.g. Colab GPUs whose
# modern triton breaks DNABERT-2's kernels). Harmless on machines without triton.
import transformers.dynamic_module_utils as _dmu
_dmu.get_imports = (lambda _o: (lambda fn: [m for m in _o(fn) if m != "triton"]))(_dmu.get_imports)
# Belt and suspenders: some transformers versions raise from check_imports via a
# path the get_imports filter above does not reach. Swallow a triton-only
# ImportError so the model still loads (DNABERT-2 falls back to torch attention).
_chk = _dmu.check_imports
def _safe_check_imports(f):
    try:
        return _chk(f)
    except ImportError:
        return _dmu.get_relative_imports(f)
_dmu.check_imports = _safe_check_imports

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from masking.collator_dam import DamageAwareDataCollator
from masking.collator_dam_offset import OffsetAwareDamageCollator
from masking.collator_mlm import MLMCollator
from masking.collator_span import SpanMaskingCollator
from masking.collator_pmdmask import PmdMaskCollator


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_collator(cfg: dict, tokenizer):
    name = cfg["collator"]
    if name == "mlm":
        return MLMCollator(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    elif name == "dam":
        # Standard DAM defaults match the original run; the extra keys drive the
        # D1 ablations purely from config (absent keys -> unchanged behaviour).
        return DamageAwareDataCollator(
            tokenizer=tokenizer,
            damage_profile_path="damage/damage_profile.npy",
            baseline_prob=cfg.get("baseline_prob", 0.0),  # A/T never masked
            scale_to=cfg.get("scale_to", 0.15),           # YAML null -> raw freqs
            smooth=cfg.get("smooth", 0),                  # raw vs smoothed profile
            power=cfg.get("power", 1.0),                  # mapping shape
        )
    elif name == "dam_offset":
        # Tokenizer-agnostic DAM (D2): per-token masking from the token's actual
        # C/G nucleotides + absolute position, not its first character.
        return OffsetAwareDamageCollator(
            tokenizer=tokenizer,
            damage_profile_path="damage/damage_profile.npy",
            scale_to=cfg.get("scale_to", 0.15),
            power=cfg.get("power", 1.0),
            smooth=cfg.get("smooth", 0),
        )
    elif name == "span":
        return SpanMaskingCollator(tokenizer=tokenizer, mlm_probability=0.15)
    elif name == "pmdmask":
        return PmdMaskCollator(
            tokenizer=tokenizer,
            damage_profile_path="damage/damage_profile.npy",
            mlm_probability=0.15,
        )
    raise ValueError(f"Unknown collator: {name}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python training/train.py <config.yaml>")
        sys.exit(1)

    cfg = load_config(sys.argv[1])
    print(f"\n=== Run: {cfg['run_name']} | Collator: {cfg['collator']} ===\n")

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    # Model loader is config-selectable. Most DNA LMs expose their masked-LM head via
    # AutoModelForMaskedLM (default). GENA-LM is the exception: its auto_map maps only
    # AutoModel -> modeling_bert.BertForMaskedLM (a pre-LayerNorm masked LM), so loading
    # it with AutoModelForMaskedLM silently falls back to the standard post-LN BERT and
    # randomly re-initialises half its LayerNorms. For GENA-LM set model_class: AutoModel.
    _model_cls = AutoModel if cfg.get("model_class") == "AutoModel" else AutoModelForMaskedLM
    model = _model_cls.from_pretrained(cfg["model_name"], trust_remote_code=True)

    # Memory-safe path for small-RAM machines (16 GB Macs on MPS, where GPU memory
    # IS system RAM): gradient checkpointing trades compute for a large cut in
    # activation memory. Best effort -- DNABERT-2's remote code may not implement
    # it, in which case small batch + the caller's MPS watermark cap are the guards.
    if cfg.get("gradient_checkpointing"):
        try:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            print("gradient checkpointing: ON")
        except Exception as e:
            print(f"gradient checkpointing unavailable ({type(e).__name__}); relying on small batch")

    # Dataset
    dataset = load_from_disk(cfg["dataset_path"])
    dataset = dataset.remove_columns(["gene"])   # Trainer expects only tensor columns

    # Collator
    collator = build_collator(cfg, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        run_name=cfg["run_name"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),  # effective batch = batch * accum
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg.get("save_total_limit"),   # cap epoch checkpoints (disk)
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        seed=cfg["seed"],
        dataloader_num_workers=cfg["dataloader_num_workers"],
        logging_steps=cfg["logging_steps"],
        report_to=cfg["report_to"],
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
    )

    trainer.train()

    # Save best checkpoint and eval results
    trainer.save_model(cfg["output_dir"])

    metrics = trainer.evaluate()
    results_path = Path(cfg["output_dir"]) / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nFinal eval loss: {metrics['eval_loss']:.4f}")
    print(f"Checkpoint saved → {cfg['output_dir']}")
    print(f"Results   saved → {results_path}")


if __name__ == "__main__":
    main()
