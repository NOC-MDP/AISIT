"""
main.py — Entry point: train and/or run inference.

Usage
-----
# Train from scratch
python main.py --mode train

# Inference on a specific time slice
python main.py --mode infer --time 2021-06

# Train then immediately infer
python main.py --mode both --time 2021-06
"""

import argparse
import logging
import random
import numpy as np
import torch
from pathlib import Path

from config import Config
from data import prepare_dataloaders
from model import TracerMLP
from train import train
from inference import infer_on_grid
from data import Normaliser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NORM_PATH = "checkpoints/normaliser.joblib"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_train(cfg: Config):
    set_seed(cfg.train.seed)

    # Data
    train_loader, val_loader, norm, n_features = prepare_dataloaders(cfg)
    norm.save(NORM_PATH)
    log.info(f"Normaliser saved to {NORM_PATH}")

    # Model
    model = TracerMLP.from_config(cfg.model, n_features)
    log.info(f"Model: {model}")

    # Train
    model, history, device = train(cfg, model, train_loader, val_loader)

    return model, norm, device


def run_infer(cfg: Config, time_slice: str):
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # Load normaliser
    norm = Normaliser.load(NORM_PATH)

    # Reconstruct model (need n_features — infer from scaler)
    n_features = norm.X_scaler.n_features_in_
    model = TracerMLP.from_config(cfg.model, n_features)

    # Load best checkpoint
    ckpt = torch.load(cfg.train.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    # Infer
    tracer_da = infer_on_grid(
        cfg,
        model,
        norm,
        device,
        time_slice  = time_slice,
        output_path = f"outputs/tracer_predicted_{time_slice or 'all'}.nc",
    )

    return tracer_da


def main():
    parser = argparse.ArgumentParser(description="Conservative tracer inference pipeline")
    parser.add_argument("--mode", choices=["train", "infer", "both"], default="both")
    parser.add_argument("--time", default=None, help="Time slice for inference, e.g. '2021-06'")
    args = parser.parse_args()

    cfg = Config()   # uses all defaults — override here or via a YAML loader

    if args.mode in ("train", "both"):
        model, norm, device = run_train(cfg)

    if args.mode in ("infer", "both"):
        tracer_da = run_infer(cfg, args.time)
        print(tracer_da)

    log.info("Done.")


if __name__ == "__main__":
    main()
