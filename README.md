# Augmenting Time Series Data with Elastic Averaging Methods

author: [Ali Ismail-Fawaz](hadifawaz1999.github.io), [@hadifawaz1999](https://github.com/hadifawaz1999)

This repository contains a python code that supports the weighted Barycenter Average method for time series data proposed by [Forestier et al. 2017 " Generating synthetic time series to augment sparse datasets" In IEEE International Conference on Data Mining (ICDM)](https://doi.org/10.1109/ICDM.2017.106). Their original proposal was to use Dynamic Time Warping Barycenter Average (DBA) [1] to generate synthetic data in order to enhance supervised learning for the task of Time Series Classification.

In this repository, we re-call the work of Forestier et al. 2017, leveraging the new time series machine learning python package, [aeon](https://github.com/aeon-toolkit/aeon), in order to use the fastest available version of elastic barycenter averaging technique.
Instead of only being able to use DBA to augment new data, we also give the user availability to choose the similarity measure of their choice, either doing DBA with DTW, or shapeDBA with shapeDTW [2], MBA with MSM [3] etc.
We add the possibility to try out the augmentation technique with a simple K-Nearest-Neighbors classifier to showcase the augmentation's add-on information to the training dataset.

In what follows, we list the required information to be able to run the code.

## Requirements

This work requires a python version `>=3.10.12`

In order to use the version of aeon that made this work possible, please install the `aif/weighted-dba` branch of the development version of aeon as follows:<br>
`pip install -U git+https://github.com/aeon-toolkit/aeon.git@aif/weighted-dba`

The following packages are installed by default with aeon: `numpy`, `scikit-learn`, `pandas`.

The rest of the packages are to be manually installed:

```
hydra-core==1.3.2
matplotlib==3.8.4
```

## Code Usage

This code utilizes the `hydra` configuration setup for the parameter selection. In order to change the parameters, simply edit the `config.yaml` file to your desire.
A detailed description of all parameters used for the augmentation method are described [in the docstring of this function](https://github.com/MSD-IRIMAS/Augmenting-TSC-Elastic-Averaging/blob/319a391bcc5b92f1a871599383ac3f021a43a036/augmentation.py#L12)

In order to run the code after setting up the configuration of parameters to your desire, simply run:<br>
`python3 main.py`

You also have the choice to edit the parameters configuration when running the above command instead of editing the configuration file, as such:<br>
```python main.py dataset="ItalyPowerDemand" distance="shape_dtw"```