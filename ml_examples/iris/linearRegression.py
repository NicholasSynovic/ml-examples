import logging
from itertools import combinations
from typing import List, Tuple

from joblib import dump
from pandas import DataFrame, Series
from progress.bar import Bar
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_examples.loaders.loadIris import load
from ml_examples.utils.utils import createBinaryClassPairings, splitData

logging.basicConfig(filename="models/linearRegression_iris.log", level=logging.INFO)


def train(
    df: DataFrame, validationDF: DataFrame
) -> Tuple[float, Tuple[str, str], LinearRegression]:
    topScore: float = 0
    bestFeatures: Tuple[str, str]
    topEstimator: LinearRegression = LinearRegression()

    columns: List[str] = df.columns.drop(labels=["Class", "EncodedLabel"]).to_list()
    columnCombinations: List[Tuple[str, str]] = list(combinations(columns, r=2))

    with Bar(
        "Trying to find the best Linear Regression model for the Iris dataset...",
        max=len(columnCombinations),
    ) as bar:
        combo: Tuple[str, str]
        for combo in columnCombinations:
            pipeline: Pipeline = make_pipeline(StandardScaler(), LinearRegression())

            trainX: Series = df[[combo[0], combo[1]]]
            trainY: Series = df["EncodedLabel"]
            validationX: Series = validationDF[[combo[0], combo[1]]]
            validationY: Series = validationDF["EncodedLabel"]

            pipeline.fit(trainX, trainY)

            score: float = pipeline.score(validationX, validationY)
            if score > topScore:
                topScore = score
                bestFeatures = combo
                topEstimator = pipeline

            bar.next()

    return (topScore, bestFeatures, topEstimator)


def main() -> None:
    topScore: float
    bestFeatures: Tuple[str, str]
    topModel: Pipeline

    trainingDF, validationDF, testingDF = splitData(df=load())

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

        dump(
            value=topModel,
            filename=f"models/linearRegression_{'_'.join(labels)}.joblib",
        )


if __name__ == "__main__":
    main()
