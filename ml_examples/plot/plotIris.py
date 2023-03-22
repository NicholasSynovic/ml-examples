from itertools import combinations
from typing import List, Tuple

from matplotlib.pyplot import savefig
from pandas import DataFrame
from progress.bar import Bar

from ml_examples.loaders.loadIris import load
from ml_examples.utils.utils import plotMultiLabeledData


def main() -> None:
    df: DataFrame = load()
    columns: List[str] = df.columns[0:4].tolist()

    columnCombinations: combinations = combinations(columns, r=2)

    with Bar(
        "Generating scatter plots of the Iris dataset...", max(len(columnCombinations))
    ) as bar:
        combo: Tuple[str, str]
        for combo in columnCombinations:
            plotMultiLabeledData(
                title="/".join(combo), df=df, xColumn=combo[0], yColumn=combo[1]
            )

            savefig(fname=f"imgs/iris_{'-'.join(combo)}.png")

            bar.next()


if __name__ == "__main__":
    main()
