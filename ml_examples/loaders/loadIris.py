from pathlib import PurePath
from typing import List

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

    columns: dict[str, str] = {
        "Sepal Length (cm)": float,
        "Sepal Width (cm)": float,
        "Petal Length (cm)": float,
        "Petal Width (cm)": float,
        "Class": str,
    }
    data: List[List[str]] = _readFile()

    df: DataFrame = DataFrame(data, columns=columns.keys())
    df.dropna(inplace=True)
    df = df.astype(columns)
    df["EncodedLabel"] = labelEncoder.fit_transform(df["Class"])

    return df
