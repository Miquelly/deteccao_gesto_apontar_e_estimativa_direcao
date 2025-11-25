import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generate_dataframe import cvs_to_df
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
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


def cross_validation():

    n_folds: int = 5
    random_state: int = 13
    epochs: int = 30
    batch_size: int = 30
    name = "banco_dados_normalizados"
    splits = 5
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

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )

    param_grid = {
        "randomforest": {
            "n_estimators": [
                100,
                200,
            ],  # Número de árvores na floresta. O artigo usa 100 e 200, e para datasets grandes, pode-se ir até 500 ou mais[cite: 200, 214].
            "max_features": [
                None,
                "sqrt",
                "log2",
                1,
                0.5,
            ],  # Número de atributos a considerar ao procurar a melhor divisão. "sqrt" é um valor comum, "log2" também é mencionado, e o artigo experimenta com F=1 (single input) e F=int(log2M+1). 'None' significa usar todos os atributos.
            "max_depth": [
                None,
                10,
                20,
                30,
            ],  # Profundidade máxima da árvore. O artigo menciona que as árvores não são podadas, o que sugere 'None' (árvores completas). No entanto, limitar a profundidade pode ser útil para controlar o overfitting e o tempo de treinamento.
            "min_samples_leaf": [
                1,
                5,
                10,
            ],  # Número mínimo de amostras necessárias para estar em um nó folha. O artigo menciona "don't split if the node size is < 5" para regressão, o que implica um valor mínimo de amostras para divisão. Para classificação, 1 é o padrão, mas valores maiores podem ajudar a suavizar o modelo.
        },
        "svc": [
            {
                "svc__kernel": ["linear"],  # Adicionado 'svc__'
                "svc__C": [0.1, 1, 10, 100],  # Adicionado 'svc__'
                "svc__probability": [True],  # Adicionado 'svc__'
            },
            {
                "svc__kernel": ["poly"],  # Adicionado 'svc__'
                "svc__degree": [2, 3],  # Adicionado 'svc__'
                "svc__gamma": ["scale", "auto", 0.001, 0.01, 0.1],  # Adicionado 'svc__'
                "svc__coef0": [0.0, 1.0],  # Adicionado 'svc__'
                "svc__C": [0.1, 1, 10, 100],  # Adicionado 'svc__'
                "svc__probability": [True],  # Adicionado 'svc__'
            },
            {
                "svc__kernel": ["rbf"],  # Adicionado 'svc__'
                "svc__gamma": [
                    "scale",
                    "auto",
                    0.001,
                    0.01,
                    0.1,
                    1,
                ],  # Adicionado 'svc__'
                "svc__C": [0.1, 1, 10, 100],  # Adicionado 'svc__'
                "svc__probability": [True],  # Adicionado 'svc__'
            },
        ],
        "logisticregression": [
            {
                # Parâmetros para regularização L1 (Lasso Regression), que promove esparsidade do modelo.
                # Solvers 'liblinear' e 'saga' são os que suportam a penalidade 'l1'.
                "penalty": ["l1"],
                "solver": ["liblinear", "saga"],
                "C": [
                    0.01,
                    0.1,
                    1,
                ],  # C é o inverso da força de regularização. Valores menores de C significam regularização mais forte.
                "max_iter": [
                    100,
                    200,
                ],  # Número máximo de iterações para os solvers convergirem.
            },
            {
                # Regularização L2 com o solver 'liblinear', que é eficaz para datasets menores.
                "penalty": ["l2"],
                "solver": ["liblinear"],
                "C": [0.01, 0.1, 1],
                "max_iter": [100, 200],
                "multi_class": ["auto"],
            },
        ],
        "decisiontree": {
            "splitter": [
                "best",
                "random",
            ],  # Estratégia para escolher a divisão em cada nó. 'best' seleciona a melhor divisão, 'random' seleciona aleatoriamente.
            "max_depth": [
                None,
                10,
                20,
                30,
                50,
            ],  # Profundidade máxima da árvore. 'None' significa que os nós são expandidos até que todas as folhas sejam puras ou contenham menos que min_samples_split amostras.
            "min_samples_split": [
                2,
                5,
                10,
                20,
            ],  # Número mínimo de amostras necessárias para dividir um nó interno.
            "min_samples_leaf": [
                1,
                2,
                4,
                8,
            ],  # Número mínimo de amostras necessárias para estar em um nó folha.
            "max_features": [
                None,
                "sqrt",
                "log2",
                0.5,
                0.8,
            ],  # Número de atributos a considerar ao procurar a melhor divisão. 'None' significa todos os atributos. 'sqrt' e 'log2' são heurísticas comuns. Valores decimais representam uma fração dos atributos.
            "class_weight": [
                None,
                "balanced",
            ],  # Pesos associados às classes. 'balanced' ajusta automaticamente os pesos inversamente proporcionais às frequências das classes, útil para datasets desbalanceados.
        },
    }

    # Modelos
    randomforest = RandomForestClassifier(
        random_state=random_state, class_weight="balanced"
    )

    decisiontree = DecisionTreeClassifier(random_state=random_state)

    logisticregression = LogisticRegression(random_state=random_state)

    svc = make_pipeline(StandardScaler(), SVC(gamma="auto"))

    modelos = [randomforest, svc, logisticregression, decisiontree]
    keys = ["randomforest", "svc", "logisticregression", "decisiontree"]

    ########################################################

    for modelo, key in zip(modelos, keys):

        print(key)

        grid_search = GridSearchCV(
            estimator=modelo,
            param_grid=param_grid[key],
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
            # scoring="accuracy",
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(Xtrain, ytrain)

        best_rf_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score_internal = grid_search.best_score_

        print(f"\n--- Resultados do Grid Search: {key} ---")
        print(f"Melhor Modelo: {best_rf_model}")
        print(f"Melhores Parâmetros: {best_params}")
        print(f"Melhor Score (Accuracy Média nos Folds): {best_score_internal:.4f}")

    ########################################################


def main():
    cross_validation()


if __name__ == "__main__":
    main()
