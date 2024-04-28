import numpy as np
import math

from typing import Dict, Optional

from sklearn.utils import check_random_state

from aeon.clustering.averaging import elastic_barycenter_average
from aeon.distances import pairwise_distance


def augment_dataset_with_elastic_barycenter(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 5,
    factor: int = 2,
    random_state: int = None,
    distance: str = "dtw",
    distance_params: Optional[dict] = None,
):
    """Function to apply data extension on a time series dataset.

    This data extension method proposed in [1]_ uses the elastic
    barycenter [2]_ on a weighted set of samples to augment a new
    sample.

    Parameters
    ----------
    X: np.ndarray
        The input dataset to be augmented of shape:
        (n_instances, n_channels, n_timepoints).
    y: np.ndarray
        The labels of the input samples of shape:
        (n_instances,).
    batch_size: int, default = 5
        The number of samples sampled from one class label
        used to generate one new sample each time.
    factor: int, default = 2
        The factor of augmentation (2, 3, 4, etc.) of X,
        if set to 2, the output will be the double of X,
        if set to 3, the output will be the triple of X, etc.
        It is obligated that factor is an integer >= 2.
    random_state: int (or None), default = None
        The random state value to control the random selection
        in the augmentation method.
    distance: str, default = "dtw"
        The distance measure used for the elastic barycenter.
        Default behavior will be using Dynamic Time Warping.
        Possible distances:
        ============================================================
        distance           name
        ============================================================
        dtw                Dynamic Time Warping
        adtw               Amerced Dynamic Time Warping
        erp                Edit Real Penalty
        edr                Edit distance for real sequences
        euclidean          Euclidean Distance
        LCSS               Longest Common Subsequence
        manhattan          Manhattan Distance
        minkowski          Minkowski Distance
        msm                Move-Split-Merge
        sbd                Shape-based Distance
        shape_dtw          Shape Dynamic Time Warping
        squared            Squared Distance
        twe                Time Warp Edit
        wddtw              Weighted Derivative Dynamic Time Warping
        wdtw               Weighted Dynamic Time Warping
        ============================================================
    distance_params: dict, default = None
        The parameters of the distance function used.

    Returns
    -------
    np.ndarray of shape (n_instances * factor, n_channels, n_timepoints)
        The output extended dataset.
    np.ndarray of shape (n_instances * factor,)
        The labels of the output extended dataset.

    References
    ----------
    .. [1] Forestier, G., Petitjean, F., Dau, H. A., Webb, G. I., & Keogh, E.
       (2017, November). Generating synthetic time series to augment sparse
       datasets. In 2017 IEEE international conference on data mining (ICDM)
       (pp. 865-870). IEEE.
    .. [2] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    """
    assert len(X) == len(y)
    assert len(X.shape) == 3
    factor = int(factor)
    assert factor >= 2

    n_instances = len(X)
    n_channels = len(X[0])
    n_timepoints = len(X[0, 0])

    new_X = np.zeros(shape=((factor - 1) * n_instances, n_channels, n_timepoints))
    new_y = np.zeros(shape=((factor - 1) * n_instances,))

    n_classes = np.unique(y)

    rng_np = np.random.default_rng(random_state)
    rng = check_random_state(random_state)

    n_generated_for_now = 0

    for c in n_classes:
        X_c = X[y == c]

        if len(X_c) == 1:
            continue
        elif len(X_c) < batch_size:
            batch_size = len(X_c)

        for _ in range((factor - 1) * len(X_c)):
            indices_batch = rng_np.choice(
                np.arange(len(X_c)), size=batch_size, replace=False
            )
            X_c_batch = X_c[indices_batch]

            new_series = _augment_batch_by_one_same_class(
                X=X_c_batch,
                random_state=rng.randint(0, np.iinfo(np.int32).max),
                distance=distance,
                distance_params=distance_params,
            )

            new_X[n_generated_for_now, :, :] = new_series
            new_y[n_generated_for_now] = c

            n_generated_for_now = n_generated_for_now + 1

    return np.concatenate((X, new_X), axis=0), np.concatenate((y, new_y), axis=0)


def _augment_batch_by_one_same_class(
    X: np.ndarray,
    random_state: int = None,
    distance: str = "dtw",
    distance_params: Optional[Dict] = None,
):

    # choose random sample in X
    rng = check_random_state(random_state)
    random_index = rng.randint(low=0, high=len(X))

    reference_series = X[random_index]

    if distance_params is not None:
        pairwise_distances_to_reference = pairwise_distance(
            x=X, y=reference_series, metric=distance, **distance_params
        )
    else:
        pairwise_distances_to_reference = pairwise_distance(
            x=X,
            y=reference_series,
            metric=distance,
        )
    neighbors_indices = np.argsort(pairwise_distances_to_reference, axis=0)

    # make sure the first index in neighbors_indices is to the ref series
    assert neighbors_indices[0] == random_index

    _distance_to_nearest = pairwise_distances_to_reference[neighbors_indices[1]]
    _distance_to_nearest = _distance_to_nearest[0, 0]

    weights = np.zeros(shape=(len(X),))

    weights[random_index] = 1.0
    weights[neighbors_indices[1]] = 0.5

    for i in range(2, len(X)):

        _distance_from_i_to_ref = pairwise_distances_to_reference[neighbors_indices[i]]
        _distance_from_i_to_ref = _distance_from_i_to_ref[0, 0]

        _weight = math.exp(
            math.log(0.5) * _distance_from_i_to_ref / _distance_to_nearest
        )

        weights[i] = _weight

    if distance_params is not None:

        _distance_params = distance_params.copy()
        del _distance_params["itakura_max_slope"]

        augmented_series = elastic_barycenter_average(
            X=X,
            distance=distance,
            init_barycenter=reference_series,
            weights=weights,
            **_distance_params
        )
    else:
        augmented_series = elastic_barycenter_average(
            X=X, distance=distance, init_barycenter=reference_series, weights=weights
        )

    return augmented_series
