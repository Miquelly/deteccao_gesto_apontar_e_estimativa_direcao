import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, decomposition
from matplotlib.ticker import MultipleLocator  # Importar MultipleLocator


def main():
    np.random.seed(5)

    dataframe = "banco_dados_normalizados"

    gesture = pd.read_csv(
        f"{dataframe}.csv",
        dtype={
            "Frame": int,
            "ID": int,
            "Gesture": int,
        },
    )

    y = gesture["Gesture"].to_numpy()

    gesture.drop(
        columns=[
            "Frame",
            "ID",
            "Gesture",
        ],
        inplace=True,
    )

    X = []

    for i in range(len(gesture)):
        p = gesture.loc[i].to_numpy()
        X.append(p)

    X = np.array(X)
    fig = plt.figure(1, figsize=(8, 6))
    plt.clf()

    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    fig.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.1)
    ax.set_position([0, 0, 0.95, 1])

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    y = np.choose(y, [0, 1]).astype(int)

    cores = np.array(["red", "blue"])
    labels = ["Non-pointing", "Pointing"]
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=cores[y.astype(int)], edgecolor="k")
    for i in np.unique(y):
        subset = X[y == i]
        ax.scatter(
            subset[:, 0],
            subset[:, 1],
            subset[:, 2],
            c=cores[i],
            label=labels[i],
            edgecolor="k",
        )

    ax.set_xlabel("x", fontsize=15)
    ax.set_ylabel("y", fontsize=15)
    ax.set_ylim([-1, 1])
    ax.set_zlabel("z", fontsize=15)
    ax.set_zlim([-1, 1])

    nome_arquivo = f"{dataframe}pca_3d_plot.png"
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.zaxis.set_major_locator(MultipleLocator(0.5))

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.95, 0.9),
        fontsize=14,
        markerscale=2,
    )
    ax.view_init(elev=38, azim=-90)
    fig.savefig(nome_arquivo, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
