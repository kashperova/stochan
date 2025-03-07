import os

import numpy as np
import plotly.graph_objects as go


def plot_spiral_pred_3d(epoch: int, pred_xyz: np.array, true_xyz: np.array, frames_dir: str, angle: int = 0) -> str:
    fig = go.Figure()

    # true spiral (blue line)
    fig.add_trace(
        go.Scatter3d(x=true_xyz[:, 0], y=true_xyz[:, 1], z=true_xyz[:, 2], mode="lines", name="True trajectory")
    )

    # learned ODE (red line)
    fig.add_trace(
        go.Scatter3d(x=pred_xyz[:, 0], y=pred_xyz[:, 1], z=pred_xyz[:, 2], mode="lines", name="Neural ODE")
    )

    # adjust camera angle
    # convert degrees to radians
    angle_rad = np.radians(angle)
    # a simple circular path in the x-y plane
    camera_eye = dict(x=2.0 * np.cos(angle_rad), y=2.0 * np.sin(angle_rad), z=1.5)

    fig.update_layout(
        title=f"",
        scene=dict(
            xaxis=dict(title="", showticklabels=False),
            yaxis=dict(title="", showticklabels=False),
            zaxis=dict(title="", showticklabels=False),
            camera=dict(eye=camera_eye)
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    filename = os.path.join(frames_dir, f"frame_{epoch:04d}.png")

    fig.write_image(filename)
    return filename
