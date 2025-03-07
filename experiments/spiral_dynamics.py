import os
import re

import hydra
import imageio
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from dynamics.spiral import train_spiral_dynamics
from models.neural_ode.neural_ode import NeuralODE
from models.spiral_ode import Spiral3dODEFunc
from utils import set_seed


def run_experiment(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    solver = instantiate(cfg.solver)
    func = Spiral3dODEFunc(ode_solver=solver).to(device)
    model = NeuralODE(func=func).to(device)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(cfg.optimizer, params=trainable_params)
    lr_scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)

    os.makedirs(cfg.frames_dir, exist_ok=True)
    frame_files = train_spiral_dynamics(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        cfg=cfg
    )

    images = [imageio.imread(f) for f in frame_files]

    duration = 5.0
    pause_frames = [images[-1]] * int(12 / duration)
    images = images + pause_frames

    imageio.mimsave(cfg.gif_filename, images, duration=duration, loop=0)
    print("Saved training GIF")

    # cleanup
    pattern = re.compile(r"^frame_\d+\.png$")
    png_frames = [f for f in os.listdir(cfg.frames_dir) if pattern.match(f)]

    for name in png_frames:
        os.remove(os.path.join(cfg.frames_dir, name))


if __name__ == "__main__":
    config_path = "../src/configs"
    config_name = "spiral_dynamics"

    hydra.initialize(config_path=config_path, version_base=None)
    config = hydra.compose(config_name=config_name)
    run_experiment(config)
