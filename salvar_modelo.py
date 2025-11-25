import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from generate_dataframe import cvs_to_df
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pickle5 import pickle
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# arquivo_joblib = "random_forest.sav"
# randomForest = joblib.load(arquivo_joblib)


def salvar_modelo():

    n_folds: int = 5
    random_state: int = 13
    epochs: int = 30
    batch_size: int = 30
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

    # DEFINIR MODELO ###############################
    modelo = "logisticRegression"
    # randomForest = RandomForestClassifier(
    #     class_weight="balanced", max_features=0.5, random_state=13
    # )

    # decisionTree = DecisionTreeClassifier(
    #     max_depth=10,
    #     max_features=0.5,
    #     min_samples_split=5,
    #     random_state=13,
    #     class_weight="balanced",
    # )

    logisticRegression = LogisticRegression(
        C=1,
        max_iter=200,
        penalty="l1",
        random_state=13,
        solver="saga",
        class_weight="balanced",
    )

    # svc = make_pipeline(
    #     StandardScaler(), SVC(C=10, gamma=1, probability=True, class_weight="balanced")
    # )
    ################################################

    # TREINO #######################################
    X_train = X
    y_train = y

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train,
    #     y_train,
    #     stratify=y_train,
    #     test_size=0.25,
    #     random_state=random_state,
    # )
    # print(len(X_train), len(y_train), len(X_test), len(y_test))

    # randomForest.fit(X_train, y_train)
    # predict = randomForest.predict(X_test)
    # folds["randomForest"][f"Fold{fold}"] = [
    #     accuracy_score(y_test, predict),
    #     precision_score(y_test, predict),
    #     recall_score(y_test, predict),
    #     f1_score(y_test, predict),
    # ]

    # decisionTree.fit(X_train, y_train, sample_weight=None, check_input=True)
    # predict = decisionTree.predict(X_test, check_input=True)
    # folds["decisionTree"][f"Fold{fold}"] = [
    #     accuracy_score(y_test, predict),
    #     precision_score(y_test, predict),
    #     recall_score(y_test, predict),
    #     f1_score(y_test, predict),
    # ]

    logisticRegression.fit(X_train, y_train)
    # predict = logisticRegression.predict(X_test)
    # folds["logisticRegression"][f"Fold{fold}"] = [
    #     accuracy_score(y_test, predict),
    #     precision_score(y_test, predict),
    #     recall_score(y_test, predict),
    #     f1_score(y_test, predict),
    # ]

    # svc.fit(X_train, y_train)
    # predict = svc.predict(X_test)
    # folds["svc"][f"Fold{fold}"] = [
    #     accuracy_score(y_test, predict),
    #     precision_score(y_test, predict),
    #     recall_score(y_test, predict),
    #     f1_score(y_test, predict),
    # ]
    ################################################

    # SALVAR MODELO ################################
    arquivo_joblib = f"{modelo}.sav"
    classificador_lr = logisticRegression
    joblib.dump(classificador_lr, arquivo_joblib)
    ################################################


def main():
    salvar_modelo()


if __name__ == "__main__":
    main()
