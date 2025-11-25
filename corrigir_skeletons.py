import os
import cv2
import json
import joblib
import warnings
import matplotlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from classificador import Gesture
from is_msgs.image_pb2 import ObjectAnnotations
from google.protobuf.json_format import ParseDict
from is_msgs.image_pb2 import HumanKeypoints as HKP

from criar_video import criar_video


# def objs_virtual(ax):
#     pts = {
#         "Robot": [-1.0, -1.5, 0.25],
#         "Umbrella": [-1.25, 1.25, 0.5],
#         "Person": [1.5, -1.0, 1.25],
#     }
#     # 045

#     chaves = list(pts.keys())

#     cores = ["r", "#277E4C", "black"]
#     i = 0

#     for obj in pts:
#         x = pts[obj][0]
#         y = pts[obj][1]
#         z = pts[obj][2]

#         ax.scatter(
#             x,
#             y,
#             zs=z,
#             linewidth=3,
#             color=cores[i],
#             label=f"{chaves[i]}",
#         )
#         i += 1

#     return pts


def objs_virtual(ax):
    pts = {
        "Robot": [-1.0, -1.5, 0.25],
        "Umbrella": [-1.25, 1.25, 0.5],
        "Person": [1.5, -1.0, 1.25],
    }
    # 045

    chaves = list(pts.keys())

    # cores = ["r", "#277E4C", "black"]
    cores = ["#0066FF", "#00FC15", "#000000"]
    i = 0

    for obj in pts:
        x = pts[obj][0]
        y = pts[obj][1]
        z = pts[obj][2]

        ax.scatter(
            x,
            y,
            zs=z,
            linewidth=8,
            color=cores[i],
            label=f"{chaves[i]}",
        )
        i += 1

    return pts


def configure_plot(ax):
    """Configurações do gráfico 3D"""
    ax.clear()
    ax.view_init(azim=0, elev=20)
    ax.set_xlim(-4, 4)
    ax.set_xticks(np.arange(-4, 4, 1))
    ax.set_ylim(-4, 4)
    ax.set_yticks(np.arange(-4, 4, 1))
    ax.set_zlim(-0.25, 2)
    ax.set_zticks(np.arange(0, 2, 0.5))
    ax.set_xlabel("X", labelpad=20)
    ax.set_ylabel("Y", labelpad=10)
    ax.set_zlabel("Z", labelpad=5)


frames = [0, 1]
for target_index in frames:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    configure_plot(ax)
    pts = objs_virtual(ax)

    ax.legend(loc="upper right", bbox_to_anchor=(1.005, 0.84), fontsize=15)

    plt.tight_layout(pad=0)

    # ---------- Plot view cv2 ---------- #
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape((height, width, 4))
    img_rgb = img[:, :, 1:4]

    output_filename = f"skeletons/sk{target_index:03d}.png"
    cv2.imwrite(output_filename, img_rgb)
    plt.close(fig)
