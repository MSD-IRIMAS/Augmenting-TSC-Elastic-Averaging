import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from aeon.datasets import load_classification
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score

from augmentation import augment_dataset_with_elastic_barycenter
from utils import znormalisation, encode_labels, create_directory, _get_distance_params


@hydra.main(config_name="config.yaml", config_path="./")
def main(args: DictConfig):
    with open("config.yaml", "w") as f:
        OmegaConf.save(args, f)

    create_directory(args.dataset + "/")
    create_directory(args.dataset + "/" + args.distance + "/")

    if os.path.exists(args.dataset + "/results.csv"):
        df = pd.read_csv(args.dataset + "/results.csv")
    else:
        df = pd.DataFrame(
            columns=["distance", "accuracy", "accuracy-da-mean", "accuracy-da-std"]
        )

    xtrain, ytrain = load_classification(name=args.dataset, split="train")
    xtest, ytest = load_classification(name=args.dataset, split="test")

    xtrain = znormalisation(xtrain)
    ytrain = encode_labels(ytrain)

    xtest = znormalisation(xtest)
    ytest = encode_labels(ytest)

    distance_params = _get_distance_params(args=args)

    knn = KNeighborsTimeSeriesClassifier(
        distance=args.distance,
        distance_params=distance_params,
        n_neighbors=1,
    )

    knn.fit(xtrain, ytrain)
    ypred = knn.predict(xtest)
    accuracy_no_augmentation = accuracy_score(
        y_true=ytest, y_pred=ypred, normalize=True
    )

    rng = check_random_state(args.random_state)

    accuracies_augmentation = []

    for _run in range(args.runs):

        _random_state = rng.randint(0, np.iinfo(np.int32).max)

        new_xtrain, new_ytrain = augment_dataset_with_elastic_barycenter(
            X=xtrain,
            y=ytrain,
            distance=args.distance,
            random_state=_random_state,
            distance_params=distance_params,
        )

        knn = KNeighborsTimeSeriesClassifier(
            distance=args.distance,
            distance_params=distance_params,
            n_neighbors=1,
        )

        knn.fit(new_xtrain, new_ytrain)
        ypred = knn.predict(xtest)
        accuracies_augmentation.append(
            accuracy_score(y_true=ytest, y_pred=ypred, normalize=True)
        )

        plt.plot(new_xtrain[0:10, 0, :].T, color="blue")
        plt.plot(new_xtrain[len(xtrain) : len(xtrain) + 10, 0, :].T, color="red")

        plt.legend(
            [Line2D([], [], lw=3, color="blue"), Line2D([], [], lw=3, color="red")],
            ["Real Samples", "Generated Samples"],
        )
        plt.savefig(
            args.dataset
            + "/"
            + args.distance
            + "/generated-series-examples-"
            + str(_run)
            + ".pdf"
        )
        
        plt.cla()

    df.loc[len(df)] = {
        "distance": args.distance,
        "accuracy": accuracy_no_augmentation,
        "accuracy-da-mean": np.mean(accuracies_augmentation),
        "accuracy-da-std": np.std(accuracies_augmentation),
    }

    df.to_csv(args.dataset + "/results.csv", index=False)


if __name__ == "__main__":
    main()
