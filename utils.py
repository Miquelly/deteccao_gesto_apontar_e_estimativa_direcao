import cv2
import json
import colorsys
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from norm import normalizacao
from classificador import Gesture
from split_clip import split_clip
from google.protobuf.json_format import Parse
from is_msgs.image_pb2 import ObjectAnnotations
from is_msgs.image_pb2 import HumanKeypoints as HKP

from config import Config

config = Config()


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


def _id_to_rgb_color(id) -> Tuple[float, float, float]:

    hue = (id % 20) / 20
    saturation = 0.8
    luminance = 0.6
    r, g, b = [x for x in colorsys.hls_to_rgb(hue, luminance, saturation)]

    return r, g, b


def load_json(filename):
    with open(file=filename, mode="r", encoding="utf-8") as file:
        options = Parse(file.read(), ObjectAnnotations())
        return options


def criar_dataframe(name: str, COLUNAS: List):
    DataFrame = pd.DataFrame(columns=COLUNAS)
    return DataFrame


def criar_csv(name: str, DataFrame):
    DataFrame.to_csv(f"{name}.csv", index=False)


def cvs_to_df(str):
    DataFrame = pd.read_csv(str)
    return DataFrame


def criar_csv_banco_dados(
    persons,
    name: str = "banco_dados_normalizados",
    normalizar: bool = True,
) -> None:

    banco_dados = criar_dataframe(name, config.COLUNAS)

    # name = "banco_dados_normalizados"
    # persons = [1, 2, 5, 14]

    # Primeiro json (Ex: p001g11_3d) >> Para cada frame, há informações das detecções (keypoints): id e posição (x, y, z).
    # Segundo json (Ex: p001g11_spots) >> Possui o intervalo de frames que foram classificados como 1 (Com gesto).
    # O arquivo split_clip é usada como função para obter/unir as informações de keypoints e classificação dos gestos em cada frame.
    # Escolha os ids das pessoas em persons = [1, 2, 5, 14], ou só descomente a lista "person_ids" no dicionário definido em setup no arquivo split_clip.py.

    for person in persons:
        skeleton_clip, spots = split_clip(person)

        for i, t in enumerate(skeleton_clip):
            classifiquer = 0

            for spot in spots:
                begin, end = spot
                if begin <= i <= end:
                    classifiquer = 1

            for sk in t:

                if normalizar:
                    skM = normalizacao(sk)
                else:
                    skM = sk

                banco_dados = banco_dados._append(
                    {
                        "Frame": i,
                        "ID": person,
                        "Gesture": classifiquer,
                        "xNOSE": skM[0][0],
                        "yNOSE": skM[0][1],
                        "zNOSE": skM[0][2],
                        "xLEFT_EYE": skM[1][0],
                        "yLEFT_EYE": skM[1][1],
                        "zLEFT_EYE": skM[1][2],
                        "xRIGHT_EYE": skM[2][0],
                        "yRIGHT_EYE": skM[2][1],
                        "zRIGHT_EYE": skM[2][2],
                        "xLEFT_EAR": skM[3][0],
                        "yLEFT_EAR": skM[3][1],
                        "zLEFT_EAR": skM[3][2],
                        "xRIGHT_EAR": skM[4][0],
                        "yRIGHT_EAR": skM[4][1],
                        "zRIGHT_EAR": skM[4][2],
                        "xLEFT_SHOULDER": skM[5][0],
                        "yLEFT_SHOULDER": skM[5][1],
                        "zLEFT_SHOULDER": skM[5][2],
                        "xRIGHT_SHOULDER": skM[6][0],
                        "yRIGHT_SHOULDER": skM[6][1],
                        "zRIGHT_SHOULDER": skM[6][2],
                        "xLEFT_ELBOW": skM[7][0],
                        "yLEFT_ELBOW": skM[7][1],
                        "zLEFT_ELBOW": skM[7][2],
                        "xRIGHT_ELBOW": skM[8][0],
                        "yRIGHT_ELBOW": skM[8][1],
                        "zRIGHT_ELBOW": skM[8][2],
                        "xLEFT_WRIST": skM[9][0],
                        "yLEFT_WRIST": skM[9][1],
                        "zLEFT_WRIST": skM[9][2],
                        "xRIGHT_WRIST": skM[10][0],
                        "yRIGHT_WRIST": skM[10][1],
                        "zRIGHT_WRIST": skM[10][2],
                        "xLEFT_HIP": skM[11][0],
                        "yLEFT_HIP": skM[11][1],
                        "zLEFT_HIP": skM[11][2],
                        "xRIGHT_HIP": skM[12][0],
                        "yRIGHT_HIP": skM[12][1],
                        "zRIGHT_HIP": skM[12][2],
                        "xLEFT_KNEE": skM[13][0],
                        "yLEFT_KNEE": skM[13][1],
                        "zLEFT_KNEE": skM[13][2],
                        "xRIGHT_KNEE": skM[14][0],
                        "yRIGHT_KNEE": skM[14][1],
                        "zRIGHT_KNEE": skM[14][2],
                        "xLEFT_ANKLE": skM[15][0],
                        "yLEFT_ANKLE": skM[15][1],
                        "zLEFT_ANKLE": skM[15][2],
                        "xRIGHT_ANKLE": skM[16][0],
                        "yRIGHT_ANKLE": skM[16][1],
                        "zRIGHT_ANKLE": skM[16][2],
                    },
                    ignore_index=True,
                )

    criar_csv(name, banco_dados)


def plot_json(
    caminho: str,
    n_frames: int,
    fig,
    ax,
    normalizar: bool = True,
):
    # caminho = f"p001g11_3d_new_test.json"
    # n_frames = 480

    data = json.load(open(caminho))
    for frame in range(3, n_frames):
        configure_plot(ax)
        skeleton: Dict = {}

        for i in range(17):
            try:
                parts = data["localizations"][frame]["objects"][0]["keypoints"][i]

            except IndexError:
                continue

            skeleton[int(parts["id"])] = [
                parts["position"]["x"],
                parts["position"]["y"],
                parts["position"]["z"],
            ]

        # sk = [
        #     skeleton[HKP.Value(config.TESTE[0])],
        #     skeleton[HKP.Value(config.TESTE[1])],
        #     skeleton[HKP.Value(config.TESTE[2])],
        #     skeleton[HKP.Value(config.TESTE[3])],
        #     skeleton[HKP.Value(config.TESTE[4])],
        #     skeleton[HKP.Value(config.TESTE[5])],
        #     skeleton[HKP.Value(config.TESTE[6])],
        #     skeleton[HKP.Value(config.TESTE[7])],
        #     skeleton[HKP.Value(config.TESTE[8])],
        #     skeleton[HKP.Value(config.TESTE[9])],
        #     skeleton[HKP.Value(config.TESTE[10])],
        #     skeleton[HKP.Value(config.TESTE[11])],
        #     skeleton[HKP.Value(config.TESTE[12])],
        #     skeleton[HKP.Value(config.TESTE[13])],
        #     skeleton[HKP.Value(config.TESTE[14])],
        #     skeleton[HKP.Value(config.TESTE[15])],
        #     skeleton[HKP.Value(config.TESTE[16])],
        # ]

        # normalizacao(sk)

        for link in config.links:
            begin, end = link

            if begin in skeleton and end in skeleton:
                x_pair = [skeleton[begin][0], skeleton[end][0]]
                y_pair = [skeleton[begin][1], skeleton[end][1]]
                z_pair = [skeleton[begin][2], skeleton[end][2]]
                color = _id_to_rgb_color(id=id)
                ax.plot(
                    x_pair,
                    y_pair,
                    zs=z_pair,
                    linewidth=3,
                    color=color,
                )

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        # print("Dimensões (width, height):", (width, height))

        # Obtém os dados no formato ARGB
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        # print("Tamanho do array antes do reshape:", img.shape)

        # Reestrutura o buffer para (altura, largura, 4)
        img = img.reshape((height, width, 4))
        # print("Shape após o reshape:", img.shape)

        # Converte de ARGB para RGB (descartando o canal alpha)
        img_rgb = img[:, :, 1:4]
        # print("Shape do array RGB:", img_rgb.shape)

        # Exibe a imagem com OpenCV
        cv2.imshow("Plot", img_rgb)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break


def plot_numpy(fig, ax):
    persons = 2
    for person in range(1, persons + 1):
        skeleton_clip, spots = split_clip(person)
        for i, t in enumerate(skeleton_clip):
            skeleton: Dict = {}
            configure_plot(ax)

            # if person != 1:
            #     continue

            for sk in t:
                # plt_reta(sk, ax)

                for link in config.links:
                    begin, end = link
                    try:

                        x_pair = [
                            sk[config.TO_COCO_IDX[begin]][0],
                            sk[config.TO_COCO_IDX[end]][0],
                        ]
                        y_pair = [
                            sk[config.TO_COCO_IDX[begin]][1],
                            sk[config.TO_COCO_IDX[end]][1],
                        ]
                        z_pair = [
                            sk[config.TO_COCO_IDX[begin]][2],
                            sk[config.TO_COCO_IDX[end]][2],
                        ]
                        color = _id_to_rgb_color(id=id)
                        ax.plot(
                            x_pair,
                            y_pair,
                            zs=z_pair,
                            linewidth=3,
                            color=color,
                        )

                    except IndexError:
                        continue

                fig.canvas.draw()
                width, height = fig.canvas.get_width_height()
                # print("Dimensões (width, height):", (width, height))

                # Obtém os dados no formato ARGB
                img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                # print("Tamanho do array antes do reshape:", img.shape)

                # Reestrutura o buffer para (altura, largura, 4)
                img = img.reshape((height, width, 4))
                # print("Shape após o reshape:", img.shape)

                # Converte de ARGB para RGB (descartando o canal alpha)
                img_rgb = img[:, :, 1:4]
                # print("Shape do array RGB:", img_rgb.shape)

                # Exibe a imagem com OpenCV
                cv2.imshow("Skeletons", img_rgb)
                key = cv2.waitKey(10)
                if key == ord("q"):
                    return


def plot_dataframe(fig, ax):

    name = "banco_dados"
    dataframe = pd.read_csv(
        f"{name}.csv",
        dtype={
            "Frame": int,
            "ID": int,
            "Gesture": int,
        },
    )

    skdict = dataframe.copy()
    skdict.drop(
        columns=[
            "ID",
            "Frame",
            "Gesture",
        ]
    )

    X = dataframe.copy()
    X.drop(
        columns=[
            "ID",
            "Frame",
            "Gesture",
            "xNOSE",
            "yNOSE",
            "zNOSE",
            "xLEFT_HIP",
            "yLEFT_HIP",
            "zLEFT_HIP",
            "xRIGHT_HIP",
            "yRIGHT_HIP",
            "zRIGHT_HIP",
            "xLEFT_EAR",
            "yLEFT_EAR",
            "zLEFT_EAR",
            "xRIGHT_EAR",
            "yRIGHT_EAR",
            "zRIGHT_EAR",
            "xLEFT_KNEE",
            "yLEFT_KNEE",
            "zLEFT_KNEE",
            "xRIGHT_KNEE",
            "yRIGHT_KNEE",
            "zRIGHT_KNEE",
            "xLEFT_ANKLE",
            "yLEFT_ANKLE",
            "zLEFT_ANKLE",
            "xRIGHT_ANKLE",
            "yRIGHT_ANKLE",
            "zRIGHT_ANKLE",
            "xLEFT_EYE",
            "yLEFT_EYE",
            "zLEFT_EYE",
            "xRIGHT_EYE",
            "yRIGHT_EYE",
            "zRIGHT_EYE",
        ],
        inplace=True,
    )

    for i in range(len(dataframe)):
        configure_plot(ax)
        for link in config.links:
            begin, end = link
            try:
                x_pair = [
                    skdict.loc[i, f"x{config.TESTE[config.TO_COCO_IDX[begin]]}"],
                    skdict.loc[i, f"x{config.TESTE[config.TO_COCO_IDX[end]]}"],
                ]
                y_pair = [
                    skdict.loc[i, f"y{config.TESTE[config.TO_COCO_IDX[begin]]}"],
                    skdict.loc[i, f"y{config.TESTE[config.TO_COCO_IDX[end]]}"],
                ]
                z_pair = [
                    skdict.loc[i, f"z{config.TESTE[config.TO_COCO_IDX[begin]]}"],
                    skdict.loc[i, f"z{config.TESTE[config.TO_COCO_IDX[end]]}"],
                ]

                color = _id_to_rgb_color(id=id)
                ax.plot(
                    x_pair,
                    y_pair,
                    zs=z_pair,
                    linewidth=3,
                    color=color,
                )

            except IndexError:
                continue

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        # print("Dimensões (width, height):", (width, height))

        # Obtém os dados no formato ARGB
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        # print("Tamanho do array antes do reshape:", img.shape)

        # Reestrutura o buffer para (altura, largura, 4)
        img = img.reshape((height, width, 4))
        # print("Shape após o reshape:", img.shape)

        # Converte de ARGB para RGB (descartando o canal alpha)
        img_rgb = img[:, :, 1:4]
        # print("Shape do array RGB:", img_rgb.shape)

        # Exibe a imagem com OpenCV
        cv2.imshow("Skeletons", img_rgb)
        key = cv2.waitKey(10)
        if key == ord("q"):
            return


def dataframe_to_parts(dfX: pd.DataFrame) -> np.ndarray:
    dfX = dfX.reset_index(drop=True)

    colunas = [
        "xLEFT_SHOULDER",
        "xRIGHT_SHOULDER",
        "xLEFT_WRIST",
        "xRIGHT_WRIST",
        "xLEFT_ELBOW",
        "xRIGHT_ELBOW",
    ]

    # xLEFT_SHOULDER, yLEFT_SHOULDER, zLEFT_SHOULDER
    idx0 = dfX.columns.get_loc(colunas[0])
    p0_L = dfX.iloc[:, idx0 : idx0 + 3].loc[0]
    p0_L = p0_L.to_numpy()

    # xRIGHT_SHOULDER, yRIGHT_SHOULDER, zRIGHT_SHOULDER
    idx1 = dfX.columns.get_loc(colunas[1])
    p0_R = dfX.iloc[:, idx1 : idx1 + 3].loc[0]
    p0_R = p0_R.to_numpy()

    # xLEFT_WRIST, yLEFT_WRIST, zLEFT_WRIST
    idx2 = dfX.columns.get_loc(colunas[2])
    p2_L = dfX.iloc[:, idx2 : idx2 + 3].loc[0]
    p2_L = p2_L.to_numpy()

    # xRIGHT_WRIST, yRIGHT_WRIST, zRIGHT_WRIST
    idx3 = dfX.columns.get_loc(colunas[3])
    p2_R = dfX.iloc[:, idx3 : idx3 + 3].loc[0]
    p2_R = p2_R.to_numpy()

    # xLEFT_ELBOW, yLEFT_ELBOW, zLEFT_ELBOW
    idx4 = dfX.columns.get_loc(colunas[4])
    p1_L = dfX.iloc[:, idx4 : idx4 + 3].loc[0]
    p1_L = p1_L.to_numpy()

    # xRIGHT_ELBOW, yRIGHT_ELBOW, zRIGHT_ELBOW
    idx5 = dfX.columns.get_loc(colunas[5])
    p1_R = dfX.iloc[:, idx5 : idx5 + 3].loc[0]
    p1_R = p1_R.to_numpy()

    # p0, pt1 >> Ombro
    # p2, pt2 >> pulso
    # p1 >> cotovelo

    parts = np.array([p0_L, p1_L, p2_L, p0_R, p1_R, p2_R])

    return parts


def main():

    obj = Gesture()

    COLUNAS = [
        "Frame",
        "ID",
        "Gesture",
        "xNOSE",
        "yNOSE",
        "zNOSE",
        "xLEFT_EYE",
        "yLEFT_EYE",
        "zLEFT_EYE",
        "xRIGHT_EYE",
        "yRIGHT_EYE",
        "zRIGHT_EYE",
        "xLEFT_EAR",
        "yLEFT_EAR",
        "zLEFT_EAR",
        "xRIGHT_EAR",
        "yRIGHT_EAR",
        "zRIGHT_EAR",
        "xLEFT_SHOULDER",
        "yLEFT_SHOULDER",
        "zLEFT_SHOULDER",
        "xRIGHT_SHOULDER",
        "yRIGHT_SHOULDER",
        "zRIGHT_SHOULDER",
        "xLEFT_ELBOW",
        "yLEFT_ELBOW",
        "zLEFT_ELBOW",
        "xRIGHT_ELBOW",
        "yRIGHT_ELBOW",
        "zRIGHT_ELBOW",
        "xLEFT_WRIST",
        "yLEFT_WRIST",
        "zLEFT_WRIST",
        "xRIGHT_WRIST",
        "yRIGHT_WRIST",
        "zRIGHT_WRIST",
        "xLEFT_HIP",
        "yLEFT_HIP",
        "zLEFT_HIP",
        "xRIGHT_HIP",
        "yRIGHT_HIP",
        "zRIGHT_HIP",
        "xLEFT_KNEE",
        "yLEFT_KNEE",
        "zLEFT_KNEE",
        "xRIGHT_KNEE",
        "yRIGHT_KNEE",
        "zRIGHT_KNEE",
        "xLEFT_ANKLE",
        "yLEFT_ANKLE",
        "zLEFT_ANKLE",
        "xRIGHT_ANKLE",
        "yRIGHT_ANKLE",
        "zRIGHT_ANKLE",
    ]

    # Ignorar avisos de depreciação
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    """Plot - Matplot"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    CASE = 1
    if CASE == 0:
        name = "banco_dados_normalizados"
        persons = [1, 2, 5, 14]  # Quais pessoas serão consideradas
        normalizar = True

        criar_csv_banco_dados(name, persons, normalizar)  # criar_gesture()

    elif CASE == 1:
        n_frames = 480
        caminho = f"p001g11_3d_new_test.json"
        normalizar = False

        plot_json(caminho, n_frames, fig, ax, normalizar)

    elif CASE == 2:
        plot_numpy(fig, ax)

    elif CASE == 3:
        plot_dataframe(fig, ax)


if __name__ == "__main__":
    main()

# /home/miquelly/Desktop/IC2/Segundo/.venv/bin/python /home/miquelly/Desktop/IC2/Segundo/generate_clip.py
# source /home/miquelly/Desktop/IC2/Segundo/.venv/bin/activate
