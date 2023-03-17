import logging
from itertools import combinations
from typing import List, Tuple

from joblib import dump
from numpy import ndarray
from pandas import DataFrame, Series
from progress.bar import Bar
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_examples.loaders.loadIris import load

logging.basicConfig(filename="models/perceptron_iris.log", level=logging.INFO)


def splitData(df: DataFrame) -> List[DataFrame]:
    trainingDF: DataFrame
    validationDF: DataFrame
    testingDF: DataFrame

    trainingDF, testingDF = train_test_split(
        df, test_size=0.15, train_size=0.85, random_state=42, shuffle=True
    )
    trainingDF, validationDF = train_test_split(
        trainingDF, test_size=0.15, train_size=0.85, random_state=42, shuffle=True
    )

    return [trainingDF, validationDF, testingDF]


def createBinaryClassPairings(df: DataFrame) -> List[DataFrame]:
    data: List[DataFrame] = []
    uniqueClasses: List[int] = df["EncodedLabel"].unique().tolist()
    classPairings: List[Tuple[int, int]] = list(combinations(uniqueClasses, r=2))

    pair: Tuple[int, int]
    for pair in classPairings:
        data.append(
            df.drop(
                df[
                    (df["EncodedLabel"] != pair[0]) & (df["EncodedLabel"] != pair[1])
                ].index
            )
        )

    return data


def train(
    df: DataFrame, validationDF: DataFrame
) -> Tuple[float, Tuple[str, str], Perceptron]:
    topScore: float = 0
    bestFeatures: Tuple[str, str]
    topEstimator: Perceptron = Perceptron()

    parameters: dict = {
        "perceptron__penalty": ("l2", "l1", "elasticnet", None),
        "perceptron__alpha": (0.0001, 0.001, 0.01, 0.1, 0.00001),
        "perceptron__l1_ratio": (0.15, 0.05, 0.25, 0.5, 0.75),
        "perceptron__max_iter": (1000, 2000, 5000, 500, 100),
    }

    pipeline: Pipeline = make_pipeline(StandardScaler(), Perceptron())

    columns: List[str] = df.columns.drop(labels=["Class", "EncodedLabel"]).to_list()
    columnCombinations: List[Tuple[str, str]] = list(combinations(columns, r=2))

    with Bar(
        "Trying to find the best Perceptron model for the Iris dataset...",
        max=len(columnCombinations),
    ) as bar:
        combo: Tuple[str, str]
        for combo in columnCombinations:
            trainX: Series = df[[combo[0], combo[1]]]
            trainY: Series = df["EncodedLabel"]
            validationX: Series = validationDF[[combo[0], combo[1]]]
            validationY: Series = validationDF["EncodedLabel"]

            gscv: GridSearchCV = GridSearchCV(pipeline, parameters)
            gscv.fit(trainX, trainY)

            model: Perceptron = gscv.best_estimator_
            score: float = model.score(validationX, validationY)
            if score > topScore:
                topScore = score
                bestFeatures = combo
                topEstimator = model

            bar.next()

    return (topScore, bestFeatures, topEstimator)


def main() -> None:
    topScore: float
    bestFeatures: Tuple[str, str]
    topModel: Perceptron

    df: DataFrame = load()

    trainingDF, validationDF, testingDF = splitData(df)

    pairs: List[DataFrame] = createBinaryClassPairings(trainingDF)

    pair: DataFrame
    for pair in pairs:
        topScore, bestFeatures, topModel = train(df=pair, validationDF=validationDF)

        labels: List[str] = pair["Class"].unique().tolist()
        logging.info(f"Evaluated Labels: {labels}")
        logging.info(f"Top Training Score: {topScore * 100}%")
        logging.info(f"Top Features: {bestFeatures}")

        X: Series = testingDF[[bestFeatures[0], bestFeatures[1]]]
        y: Series = testingDF["EncodedLabel"]
        score: float = topModel.score(X, y)

        logging.info(f"Testing Score: {score * 100}%\n")

        dump(value=topModel, filename=f"models/perceptron_{'_'.join(labels)}.joblib")


if __name__ == "__main__":
    main()
