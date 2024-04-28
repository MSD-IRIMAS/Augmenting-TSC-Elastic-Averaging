import numpy as np
import os

from sklearn.preprocessing import LabelEncoder


def create_directory(directory_path):
    """Create a non-existing directory.

    Parameters
    ----------
    directory_path: str
        The directory to be created if non existing.

    Returns
    -------
    None.
    """
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)


def encode_labels(y):
    """Encode labels.

    Parameters
    ----------
    y: np.ndarray
        The input labels of shape (n_instances)

    Returns
    -------
    np.ndarray of shape (n_instances)
        The output labels encoded using sklearn.
    """
    labenc = LabelEncoder()
    return labenc.fit_transform(y)


def znormalisation(x):
    """Z-Normalize the input time series on the time axis.

    Parameters
    ----------
    x: np.ndarray
        The input time series dataset of shape:
        (n_instances, n_channels, n_timepoints).

    Returns
    -------
    np.ndarray of shape (n_instances, n_channels, n_timepoints)
        The z-normalized version of x on the time axis.
    """
    stds = np.std(x, axis=2, keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=2, keepdims=True)) / stds
    return (x - x.mean(axis=2, keepdims=True)) / (x.std(axis=2, keepdims=True))


def _get_distance_params(args):

    distance_params = {}

    if args.distance == "dtw" or args.distance == "ddtw":
        distance_params["window"] = args.distance_params.window
        distance_params["itakura_max_slope"] = args.distance_params.itakura_max_slope
    elif args.distance == "shape_dtw":
        distance_params["window"] = args.distance_params.window
        distance_params["itakura_max_slope"] = args.distance_params.itakura_max_slope
        distance_params["descriptor"] = args.distance_params.descriptor
        distance_params["reach"] = args.distance_params.reach
    elif args.distance == "msm":
        distance_params["window"] = args.distance_params.window
        distance_params["itakura_max_slope"] = args.distance_params.itakura_max_slope
        distance_params["independent"] = args.distance_params.independent
        distance_params["c"] = args.distance_params.c
        
        
        
    return distance_params