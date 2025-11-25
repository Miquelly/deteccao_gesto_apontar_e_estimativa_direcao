import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from generate_dataframe import cvs_to_df
from sklearn.pipeline import make_pipeline
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


def generate_and_save_table_image(
    model_name, model_data, metrics, output_filename=None
):
    """
    Gera uma tabela de métricas de validação cruzada para um modelo específico
    e tenta salvá-la como uma imagem.

    Args:
        model_name (str): O nome do modelo (ex: "randomForest").
        model_data (dict): O dicionário de dados para o modelo (contém FoldX e Average).
        metrics (list): Lista de strings com os nomes das métricas.
        output_filename (str, optional): O nome do arquivo para salvar a imagem.
                                         Se None, a imagem será exibida (se em ambiente interativo).
    """
    # Preparar os dados para o DataFrame do Pandas
    rows = []
    index_labels = []

    # Adicionar os dados dos folds
    for i in range(1, 6):  # Folds 1 a 5
        fold_key = f"Fold{i}"
        if fold_key in model_data and len(model_data[fold_key]) == len(metrics):
            rows.append(model_data[fold_key])
            index_labels.append(f"Fold {i}")
        else:
            print(
                f"Aviso: Dados incompletos ou ausentes para {model_name} - {fold_key}. Pulando."
            )
            # Se for crucial ter 5 folds, você pode querer preencher com NaNs ou levantar um erro

    # Adicionar os dados da Média
    average_key = "Average"
    if average_key in model_data and len(model_data[average_key]) == len(metrics):
        rows.append(model_data[average_key])
        index_labels.append("Average")
    else:
        print(
            f"Aviso: Dados incompletos ou ausentes para {model_name} - {average_key}. Pulando."
        )

    # Criar o DataFrame
    df = pd.DataFrame(rows, index=index_labels, columns=metrics)

    # Formatar os valores para exibição (arredondar e percentual para Accuracy)
    # df['Accuracy'] = (df['Accuracy'] * 100).round(4)
    # As outras métricas já estão em 0-1, arredondar para 2 casas decimais
    df[metrics] = df[metrics].round(4)

    # Para a coluna 'Accuracy', você pode querer formatar com duas casas decimais e o sinal de %
    # Mas a imagem original mostra como número. Vamos manter como número e 2 casas decimais.
    df["Accuracy"] = (df["Accuracy"] * 100).round(4)  # Multiplicar por 100 e arredondar

    # Para Precision, Recall, F1 score: arredondar para 2 casas decimais
    for col in ["Precision", "Recall", "F1 score"]:
        df[col] = (df[col] * 100).round(4)

    print(f"\n--- Tabela para {model_name} ---")
    print(df)
    print("--------------------------------\n")

    # --- Gerar a imagem da tabela usando Matplotlib ---
    fig, ax = plt.subplots(figsize=(8, 4))  # Ajuste o tamanho conforme necessário
    ax.axis("off")  # Remove os eixos
    ax.axis("tight")  # Ajusta o layout para a tabela

    # Criar a tabela Matplotlib
    # Formatação especial para a linha 'Average' (negrito)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Tamanho da fonte

    # Estilizar o cabeçalho
    for (row, col), cell in table._cells.items():
        if row == 0:  # Cabeçalho da coluna
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d3d3d3")  # Cor de fundo cinza claro para o cabeçalho
        if row == len(df.index):  # Última linha (Average)
            cell.set_text_props(weight="bold")
            cell.set_facecolor(
                "#f0f0f0"
            )  # Cor de fundo levemente diferente para Average

    # Ajustar o layout da tabela para que caiba no Axes
    table.auto_set_column_width(col=list(range(len(df.columns))))

    plt.title(
        f"Performance of {model_name.replace('logisticRegression', 'Logistic Regression').replace('randomForest', 'Random Forest').replace('decisionTree', 'Decision Tree').replace('svc', 'SVC')}",
        fontsize=14,
        weight="bold",
        pad=20,
    )  # Título da tabela
    plt.tight_layout()  # Ajusta o layout para evitar sobreposição

    if output_filename:
        # Salva a figura como uma imagem
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        plt.close(fig)  # Fecha a figura para não ocupar memória
        print(f"Tabela para {model_name} salva como {output_filename}")
    else:
        plt.show()  # Exibe a figura (útil em notebooks)


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
            "Fold1": [],
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
        # print(len(idx_train))

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

        decisionTree.fit(X_train, y_train, sample_weight=None, check_input=True)
        predict = decisionTree.predict(X_test, check_input=True)
        folds["decisionTree"][f"Fold{fold}"] = [
            accuracy_score(y_test, predict),
            precision_score(y_test, predict),
            recall_score(y_test, predict),
            f1_score(y_test, predict),
        ]

        logisticRegression.fit(X_train, y_train)
        predict = logisticRegression.predict(X_test)
        folds["logisticRegression"][f"Fold{fold}"] = [
            accuracy_score(y_test, predict),
            precision_score(y_test, predict),
            recall_score(y_test, predict),
            f1_score(y_test, predict),
        ]

        svc.fit(X_train, y_train)
        predict = svc.predict(X_test)
        folds["svc"][f"Fold{fold}"] = [
            accuracy_score(y_test, predict),
            precision_score(y_test, predict),
            recall_score(y_test, predict),
            f1_score(y_test, predict),
        ]

        print(f1_score(y_test, predict)),
        print(
            (2 * precision_score(y_test, predict) * recall_score(y_test, predict))
            / (precision_score(y_test, predict) + recall_score(y_test, predict))
        )
        ################################################

    # GERAR TABELAS ################################
    for model_name, model_data in folds.items():
        average = np.array([0, 0, 0, 0])
        for fold_name, fold_list in model_data.items():
            average = average + np.array(fold_list)

        average = average / 4
        folds[f"{model_name}"]["Average"] = average.tolist()

    data = folds
    metrics = ["Accuracy", "Precision", "Recall", "F1 score"]

    for model_name, model_data in data.items():
        clean_model_name = (
            model_name.replace("randomForest", "random_forest")
            .replace("decisionTree", "decision_tree")
            .replace("logisticRegression", "logistic_regression")
            .replace("svc", "svc")
        )
        output_filename = f"table_{clean_model_name}.png"
        generate_and_save_table_image(model_name, model_data, metrics, output_filename)
    ################################################


if __name__ == "__main__":
    main()
