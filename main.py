import cv2
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from classificador import Gesture

from config import Config
import utils

config = Config()


def plt_reta(obj, sk: np.ndarray, ax, y: int = 0, option: bool = False):
    (
        rightx,
        righty,
        rightz,
        leftx,
        lefty,
        leftz,
        p2_L,
        left_vector,
        p2_R,
        right_vector,
    ) = obj.reta_para_plot(sk)

    if not option:
        (
            y,
            left_distancia,
            left_teta,
            right_distancia,
            right_teta,
            left_verticalidade,
            right_verticalidade,
        ) = obj.classificador3(sk, 0.30, 30)

        classificacao = y

        if classificacao == 3:
            ax.plot(
                rightx,
                righty,
                rightz,
                color="m",
            )

            ax.plot(
                leftx,
                lefty,
                leftz,
                color="b",
            )

        elif classificacao == 1:
            ax.plot(
                rightx,
                righty,
                rightz,
                color="m",
            )

        elif classificacao == 2:
            ax.plot(
                leftx,
                lefty,
                leftz,
                color="b",
            )

    if option and classificacao == 1:
        ax.plot(
            rightx,
            righty,
            rightz,
            color="m",
        )

        ax.plot(
            leftx,
            lefty,
            leftz,
            color="b",
        )


def modelo_classificador(
    arquivo_modelo: str,
    name_dataframe: str,
    fig,
    ax,
):
    obj = Gesture()
    # arquivo_modelo = "logisticRegression.sav"
    randomForest = joblib.load(arquivo_modelo)

    # name = "gesture_ufes_norm_ok"
    dataframe = pd.read_csv(
        f"{name_dataframe}.csv",
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
        utils.configure_plot(ax)
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

                id = 0
                color = utils._id_to_rgb_color(id=id)
                ax.plot(
                    x_pair,
                    y_pair,
                    zs=z_pair,
                    linewidth=3,
                    color=color,
                )

            except IndexError:
                continue

        parts = utils.dataframe_to_parts(X.loc[[i]])

        predict_modelo = randomForest.predict(X.loc[[i]])

        plt_reta(obj, parts, ax, predict_modelo, False)

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


def main():

    obj = Gesture()

    # Ignorar avisos de depreciação
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    """Plot - Matplot"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    normalizar = True

    arquivo_modelo = "logisticRegression.sav"
    name_dataframe = "banco_dados_normalizados"  # Já deve estar normalizado
    modelo_classificador(arquivo_modelo, name_dataframe, fig, ax)


if __name__ == "__main__":
    main()

# /home/miquelly/Desktop/IC2/Segundo/.venv/bin/python /home/miquelly/Desktop/IC2/Segundo/generate_clip.py
# source /home/miquelly/Desktop/IC2/Segundo/.venv/bin/activate
