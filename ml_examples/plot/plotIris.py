from itertools import combinations
from typing import List, Tuple

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame

from ml_examples.loaders.loadIris import load


def main() -> None:
    df: DataFrame = load()
    columns: List[str] = df.columns[0:4].tolist()

    columnCombinations: combinations = combinations(columns, r=2)

    combo: Tuple[str, str]
    for combo in columnCombinations:
        titleStr: str = " / ".join(combo)

        xLimit: List[float] = [
            float(df[combo[0]].min()) - 1,
            float(df[combo[0]].max()) + 1,
        ]

        yLimit: List[float] = [
            float(df[combo[1]].min()) - 1,
            float(df[combo[1]].max()) + 1,
        ]

        ax: Axes = df.plot(
            x=combo[0],
            y=combo[1],
            kind="scatter",
            title=titleStr,
            legend=True,
            xlabel=combo[0],
            ylabel=combo[1],
        )

        figure: Figure = ax.get_figure()
        figure.savefig(fname=f"iris_{''.join(combo)}.png")


if __name__ == "__main__":
    main()
