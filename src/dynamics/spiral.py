import torch
from omegaconf import DictConfig

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn import functional as F
from tqdm import tqdm

from models.neural_ode.neural_ode import NeuralODE
from utils.plots import plot_spiral_pred_3d


def get_3d_spiral(t_size: int, t_n_steps: int) -> tuple[Tensor, Tensor]:
    t = torch.linspace(0, t_size, t_n_steps, dtype=torch.float32)
    x = torch.exp(t / 10.0) * torch.cos(2.0 * t)
    y = torch.exp(t / 10.0) * torch.sin(2.0 * t)
    return torch.stack([x, y, t], dim=1), t


def train_spiral_dynamics(
    model: NeuralODE,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    device: torch.device,
    cfg: DictConfig,
) -> list[str]:
    spiral, time_steps = get_3d_spiral(t_size=cfg.t_size, t_n_steps=cfg.t_n_steps)
    z0 = spiral[0].unsqueeze(0).to(device)  # initial state (batch=1, dims=3)
    frame_files = []

    model.train()
    pbar = tqdm(range(cfg.n_epochs), desc="Training")
    for epoch in pbar:
        optimizer.zero_grad()
        # predict the entire trajectory
        z_pred = model(z0, t=time_steps, return_trajectory=True)  # (time_len, 1, 3)
        z_pred = z_pred.squeeze(1)  # (time_len, 3)

        loss = F.mse_loss(z_pred, spiral)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        pbar.set_postfix({'Epoch': epoch, 'Loss': f"{loss.item():.6f}"})

        # generate frame for GIF every few epochs
        if epoch % cfg.save_freq == 0 or epoch == (cfg.n_epochs - 1):
            pred_xyz = z_pred.detach().cpu().numpy()
            angle = 1 * epoch  # rotate 1° per epoch
            frame_files.append(
                plot_spiral_pred_3d(
                    epoch,
                    pred_xyz,
                    spiral.cpu().numpy(),
                    frames_dir=cfg.frames_dir,
                    angle=angle,
                )
            )

    return frame_files
