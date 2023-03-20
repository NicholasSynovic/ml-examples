from itertools import combinations
from typing import List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pandas import DataFrame

from ml_examples.loaders.loadIris import load


def main() -> None:
    validColors: List[str] = list(mcolors.BASE_COLORS.values())

    df: DataFrame = load()
    columns: List[str] = df.columns[0:4].tolist()
    uniqueClasses: List[str] = df["Class"].unique().tolist()

    columnCombinations: combinations = combinations(columns, r=2)

    colors: dict[str, int] = {
        key: value
        for key, value in zip(uniqueClasses, validColors[0 : len(uniqueClasses)])
    }

    combo: Tuple[str, str]
    for combo in columnCombinations:
        plt.title(f"{combo[0]} / {combo[1]}")
        plt.scatter(
            df[combo[0]], df[combo[1]], c=df["Class"].apply(lambda x: colors[x])
        )

        handles: List[Line2D] = [
            plt.plot([], [], color=color, marker="o", ls="", markersize=10)[0]
            for color in colors.values()
        ]

        labels: List[str] = list(colors.keys())

        plt.xlabel(combo[0])
        plt.ylabel(combo[1])

        plt.legend(handles, labels, loc="upper right")
        plt.savefig(fname=f"iris_{''.join(combo)}.png")


if __name__ == "__main__":
    main()
