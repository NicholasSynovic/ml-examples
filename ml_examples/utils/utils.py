from itertools import combinations
from typing import List, Tuple

from pandas import DataFrame
from sklearn.model_selection import train_test_split


def splitData(
    df: DataFrame,
    testSize: float = 0.15,
    trainSize: float = 0.7,
    validationSize: float = 0.15,
    seed: int = 42,
) -> List[DataFrame]:
    trainingDF: DataFrame
    validationDF: DataFrame
    testingDF: DataFrame

    trainingDF, testingDF = train_test_split(
        df,
        test_size=testSize,
        train_size=trainSize + validationSize,
        random_state=seed,
        shuffle=True,
    )
    trainingDF, validationDF = train_test_split(
        trainingDF,
        test_size=validationSize,
        train_size=trainSize,
        random_state=seed,
        shuffle=True,
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
