# AdaptiveESM: Towards Adaptive Experience Sampling of Emotions in the Wild with Active Learning
This repository contains proof-of-concept codes implementing an adaptive sampling method based on active learning for collecting emotions in the wild.
---

## Motivation
An *Adaptive Experience Sampling Method (ESM)* aims to address both issues of the timeliness of queries and the quality of collected data with an algorithm that automatically determines when to issue queries to users. Well-times questionnaires are critical for sampling emotions in the wild as emotions can be subject to cognitive biases in retrospect, and the distribution of naturally occurring emotions are inherently biased.
---

## Implementation
Towards this goal, binary classifiers are trained to distinguish between low and high classes for both arousal and valence, given physiological signals collected with mobile wearable sensors in the context of paired debates as inputs.

### Dependencies and requirements
The current version depends on the following python packages:
* [pytorch >= 1.6.0](https://pytorch.org/)[pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [xgboost](https://github.com/dmlc/xgboost)
* [scipy](https://www.scipy.org/), [numpy](https://numpy.org/), [sklearn](https://scikit-learn.org/stable/), [pandas](https://pandas.pydata.org/), [tqdm](https://github.com/tqdm/tqdm)
* [pyteap](https://github.com/cheulyop/PyTEAP)

Also, codes in this repository have been tested with the [K-EmoCon dataset](https://www.nature.com/articles/s41597-020-00630-y). You need access to the dataset to replicate experiments, which you can make a request on [Zenodo](https://doi.org/10.5281/zenodo.3931963).

### Usage
Begin with cloning this repository to use codes in this repo.
```console
$ git clone https://github.com/cheulyop/AdaptiveESM.git
$ cd AdaptiveESM
```

Then, run `experiment.py` with a path to an experiment configuration file set with a `--config` argument. For example, to run an active learning experiment with XGBoost:
```console
$ python experiment.py --config configs/xgb_active_holdout.json
```

For the specifications of an experiment, refer to configuration files in the `configs` folder.
