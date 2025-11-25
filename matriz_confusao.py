import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from generate_dataframe import cvs_to_df
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():

    # BANCO DADOS ##################################
    name = "banco_dados_normalizados"
    df = cvs_to_df(f"{name}.csv")

    X = df.drop(
        columns=[
            "Gesture",
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
        ]
    )
    y = df["Gesture"]
    ################################################

    # ESTRUTURA TABELAS ############################
    folds = {
        "randomForest": {
            "Fold1": [],
            "Fold2": [],
            "Fold3": [],
            "Fold4": [],
            "Average": [0, 0, 0, 0],
        },
        "decisionTree": {
            "Fold1": [1],
            "Fold2": [],
            "Fold3": [],
            "Fold4": [],
            "Average": [0, 0, 0, 0],
        },
        "logisticRegression": {
            "Fold1": [],
            "Fold2": [],
            "Fold3": [],
            "Fold4": [],
            "Average": [0, 0, 0, 0],
        },
        "svc": {
            "Fold1": [],
            "Fold2": [],
            "Fold3": [],
            "Fold4": [],
            "Average": [0, 0, 0, 0],
        },
    }
    ################################################

    # MODELOS ######################################
    randomForest = RandomForestClassifier(
        class_weight="balanced", max_features=0.5, random_state=13
    )

    decisionTree = DecisionTreeClassifier(
        max_depth=10,
        max_features=0.5,
        min_samples_split=5,
        random_state=13,
        class_weight="balanced",
    )

    logisticRegression = LogisticRegression(
        C=1,
        max_iter=200,
        penalty="l1",
        random_state=13,
        solver="saga",
        class_weight="balanced",
    )

    svc = make_pipeline(
        StandardScaler(), SVC(C=10, gamma=1, probability=True, class_weight="balanced")
    )
    ################################################

    # MATRIZ CONFUSAO ##############################
    cm_logisticregression = []
    cm_randomforest = []
    cm_decisiontree = []
    cm_svc = []
    ################################################

    # IDX PERSONS ##################################
    idx_person0 = (0, df["ID"].value_counts()[1.0] - 1)
    idx_person1 = (
        df["ID"].value_counts()[1.0],
        df["ID"].value_counts()[1.0] + df["ID"].value_counts()[2.0] - 1,
    )
    idx_person2 = (idx_person1[1] + 1, idx_person1[1] + df["ID"].value_counts()[5.0])
    idx_person3 = (
        idx_person2[1] + 1,
        idx_person2[1] + df["ID"].value_counts()[14.0],
    )
    ################################################

    # CROSS VALIDATION #############################
    number_persons = 4

    for i in range(number_persons):

        # FOLDS PERSON #################################
        fold = i + 1
        skf_persons = [
            list(range(idx_person0[0], idx_person0[1])),
            list(range(idx_person1[0], idx_person1[1])),
            list(range(idx_person2[0], idx_person2[1])),
            list(range(idx_person3[0], idx_person3[1])),
        ]

        idx_test = skf_persons[i]

        skf_persons.pop(i)
        idx_train = skf_persons[0] + skf_persons[1] + skf_persons[2]
        ################################################

        X_train = X.iloc[idx_train]
        X_test = X.iloc[idx_test]

        y_train = y.iloc[idx_train]
        y_test = y.iloc[idx_test]

        # X_train, X_val, y_train, y_val = train_test_split(
        #     X_train,
        #     y_train,
        #     stratify=y_train,
        #     test_size=0.25,
        #     random_state=random_state,
        # )
        # print(len(X_train), len(y_train), len(X_test), len(y_test))

        randomForest.fit(X_train, y_train)
        predict = randomForest.predict(X_test)
        folds["randomForest"][f"Fold{fold}"] = [
            accuracy_score(y_test, predict),
            precision_score(y_test, predict),
            recall_score(y_test, predict),
            f1_score(y_test, predict),
        ]

        cm_randomforest.append(confusion_matrix(y_test, predict))

        decisionTree.fit(X_train, y_train, sample_weight=None, check_input=True)
        predict = decisionTree.predict(X_test, check_input=True)
        folds["decisionTree"][f"Fold{fold}"] = [
            accuracy_score(y_test, predict),
            precision_score(y_test, predict),
            recall_score(y_test, predict),
            f1_score(y_test, predict),
        ]

        cm_decisiontree.append(confusion_matrix(y_test, predict))

        logisticRegression.fit(X_train, y_train)
        predict = logisticRegression.predict(X_test)
        folds["logisticRegression"][f"Fold{fold}"] = [
            accuracy_score(y_test, predict),
            precision_score(y_test, predict),
            recall_score(y_test, predict),
            f1_score(y_test, predict),
        ]

        cm_logisticregression.append(confusion_matrix(y_test, predict))

        svc.fit(X_train, y_train)
        predict = svc.predict(X_test)
        folds["svc"][f"Fold{fold}"] = [
            accuracy_score(y_test, predict),
            precision_score(y_test, predict),
            recall_score(y_test, predict),
            f1_score(y_test, predict),
        ]

        cm_svc.append(confusion_matrix(y_test, predict))
        ################################################

    # CONFUSAO MATRIZ SEPARADA #####################
    # .append(confusion_matrix(y_test, predict, normalize="pred"))
    # display_unico = [
    #     cm_logisticregression,
    #     cm_randomforest,
    #     cm_decisiontree,
    #     cm_svc,
    # ]

    # for modelo in display_unico:
    #     for matriz in modelo:
    #         fig, ax = plt.subplots(figsize=(6, 6))

    #         disp = ConfusionMatrixDisplay(matriz, display_labels=["0", "1"])

    #         disp.plot(cmap=plt.cm.Blues, ax=ax)

    #         plt.show()

    #         plt.close(fig)

    ################################################

    # MÉDIA NORMALIZADA PRED #######################
    # .append(confusion_matrix(y_test, predict))
    cm_logisticregression = np.mean(cm_logisticregression, axis=0)
    cm_randomforest = np.mean(cm_randomforest, axis=0)
    cm_decisiontree = np.mean(cm_decisiontree, axis=0)
    cm_svc = np.mean(cm_svc, axis=0)

    somas_colunas = np.sum(cm_logisticregression, axis=0, keepdims=True)
    cm_logisticregression = cm_logisticregression / somas_colunas

    somas_colunas = np.sum(cm_randomforest, axis=0, keepdims=True)
    cm_randomforest = cm_randomforest / somas_colunas

    somas_colunas = np.sum(cm_decisiontree, axis=0, keepdims=True)
    cm_decisiontree = cm_decisiontree / somas_colunas

    somas_colunas = np.sum(cm_svc, axis=0, keepdims=True)
    cm_svc = cm_svc / somas_colunas

    display = [
        ConfusionMatrixDisplay(cm_logisticregression, display_labels=["0", "1"]),
        ConfusionMatrixDisplay(cm_randomforest, display_labels=["0", "1"]),
        ConfusionMatrixDisplay(cm_decisiontree, display_labels=["0", "1"]),
        ConfusionMatrixDisplay(cm_svc, display_labels=["0", "1"]),
    ]

    rotulo = [
        "logisticregression",
        "randomforest",
        "decisiontree",
        "svc",
    ]

    for i in range(len(display)):
        fig, ax = plt.subplots(figsize=(6, 6))

        ax.tick_params(labelsize=25)
        ax.tick_params(labelsize=25)

        ax.set_xlabel("Predicted Label", fontsize=25)
        ax.set_ylabel("True Label", fontsize=25)

        disp = display[i]
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)

        if hasattr(disp, "im_") and disp.im_ is not None:
            disp.im_.set_clim(vmin=0.0, vmax=1.0)  # <--- MUDANÇA CRÍTICA AQUI!

        for text in disp.text_.ravel():
            text.set_fontsize(25)

        cbar = fig.colorbar(disp.im_, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=20)

        plt.savefig(
            f"pred_aumento_matriz_confusao_media_{rotulo[i]}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()

        plt.close(fig)
    ################################################

    # MÉDIA NORMALIZADA ALL ########################
    # # .append(confusion_matrix(y_test, predict))
    # cm_logisticregression = np.mean(cm_logisticregression, axis=0)
    # cm_randomforest = np.mean(cm_randomforest, axis=0)
    # cm_decisiontree = np.mean(cm_decisiontree, axis=0)
    # cm_svc = np.mean(cm_svc, axis=0)

    # cm_logisticregression = cm_logisticregression / np.sum(cm_logisticregression)
    # cm_randomforest = cm_randomforest / np.sum(cm_randomforest)
    # cm_decisiontree = cm_decisiontree / np.sum(cm_decisiontree)
    # cm_svc = cm_svc / np.sum(cm_svc)

    # display = [
    #     ConfusionMatrixDisplay(cm_logisticregression, display_labels=["0", "1"]),
    #     ConfusionMatrixDisplay(cm_randomforest, display_labels=["0", "1"]),
    #     ConfusionMatrixDisplay(cm_decisiontree, display_labels=["0", "1"]),
    #     ConfusionMatrixDisplay(cm_svc, display_labels=["0", "1"]),
    # ]

    # rotulo = [
    #     "logisticregression",
    #     "randomforest",
    #     "decisiontree",
    #     "svc",
    # ]

    # for i in range(len(display)):
    #     fig, ax = plt.subplots(figsize=(6, 6))

    #     disp = display[i]
    #     disp.plot(cmap=plt.cm.Blues, ax=ax)

    #     plt.savefig(
    #         f"matriz_confusao_media_{rotulo[i]}.png", dpi=300, bbox_inches="tight"
    #     )

    #     plt.show()

    #     plt.close(fig)
    ################################################


if __name__ == "__main__":
    main()
