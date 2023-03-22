from itertools import combinations
from typing import List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pandas import DataFrame, Series
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


def plotMultiLabeledData(
    title: str, df: DataFrame, xColumn: str, yColumn: str, labelColumn: str = "Class"
) -> None:
    labels: Series = df[labelColumn]
    uniqueClasses: List[str] = df[labelColumn].unique().tolist()
    validColors: List[str] = list(mcolors.BASE_COLORS.values())

    colors: dict[str, int] = {
        key: value
        for key, value in zip(uniqueClasses, validColors[0 : len(uniqueClasses)])
    }

    plt.title(title)
    plt.scatter(df[xColumn], df[yColumn], c=labels.apply(lambda x: colors[x]))

    handles: List[Line2D] = [
        plt.plot([], [], color=color, marker="o", ls="", markersize=10)[0]
        for color in colors.values()
    ]

    labels: List[str] = list(colors.keys())

    plt.xlabel(xColumn)
    plt.ylabel(yColumn)

    plt.legend(handles, labels, loc="upper right")
