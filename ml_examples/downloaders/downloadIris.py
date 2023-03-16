from pathlib import PurePath

from requests import Response, get

dataURL: str = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)


def main() -> None:
    dataPath: PurePath = PurePath("../data/iris.data")

    dataResp: Response = get(dataURL)

    with open(dataPath, mode="wb") as dataFile:
        dataFile.write(dataResp.content)
        dataFile.close()


if __name__ == "__main__":
    main()
