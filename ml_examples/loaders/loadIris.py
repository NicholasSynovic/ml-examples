from pathlib import PurePath
from typing import Any, List

from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

dataPath: PurePath = PurePath("../data/iris.data")


def _readFile() -> List[List[str]]:
    data: List[List[str]] = []
    with open(dataPath, mode="r") as dataFile:
        lines: List[str] = dataFile.readlines()
        dataFile.close()

    line: str
    for line in lines:
        foo: List[str] = line.strip().split(sep=",")
        data.append(foo)

    return data


def load() -> DataFrame:
    labelEncoder: LabelEncoder = LabelEncoder()

    columns: List[str] = [
        "Sepal Length (cm)",
        "Sepal Width (cm)",
        "Petal Length (cm)",
        "Petal Width (cm)",
        "Class",
    ]
    data: List[List[str]] = _readFile()
    df: DataFrame = DataFrame(data, columns=columns)
    df.dropna(inplace=True)

    df["EncodedLabel"] = labelEncoder.fit_transform(df["Class"])

    return df
