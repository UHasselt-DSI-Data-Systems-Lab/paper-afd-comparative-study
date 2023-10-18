# AFD measures

A collection of measures for Approximate Functional Dependencies in relational data. Additionally, this repository contains all artifacts to "Approximately Measuring Functional Dependencies: a Comparative Study".

## Short description

In real-world research projects, we often encounter unknown relational (tabular)
datasets. In order to process them efficiently, functional dependencies (FDs) give us structural
insight into relational data, describing strong relationships between columns. Errors in real-world
data let traditional FD detection techniques fail. Hence we consider approximate FDs (AFDs): FDs
that approximately hold in relational data. 

This repository contains the implemented measures as well as the all artifacts to "Approximately Measuring Functional Dependencies: a Comparative Study".

## Overview

* `code`: this directory holds the code used to generate the results in the paper
	* `afd_measures`: all Python source code relating to the implemented AFD measures
	* `experiments`: Jupyter notebooks containing the processing steps to generate the results, figures or tables in the paper
	* `synthetic_data`: all Python source code relating to the synthetic data generation process
* `data`: the datasets used in the paper
	* `rwd`: manually annotated dataset of files found on the web (see `data/ground_truth.csv`)
	* `rwd_e`: datasets from `rwd` with errors introduced into them. Generated by the notebook `code/experiments/create_rwd_e_dataset.ipynb`.
	* `syn_e`: synthetic dataset generated focussing on errors. Generated by the notebook `code/experiments/create_syn_e.ipynb`
	* `syn_u`: synthetic dataset generated focussing on left-hand side uniqueness. Generated by the notebook `code/experiments/create_syn_u.ipynb`
	* `syn_s`: synthetic dataset generated focussing on right-hand side skewness. Generated by the notebook `code/experiments/create_syn_s.ipynb`
* `paper`: A full version of the paper including all proofs.
* `results`: results of applying the AFD measures to the datasets.

## Installation (measure library)

This library can be found on [PyPI](https://pypi.org): [`afd-measures`](https://pypi.org/project/afd-measures). Install it using `pip` like this:

```sh
pip install afd-measures
```

### Usage (measure library)

To apply one of the measures to your data, you will need a pandas DataFrame of your relation. Pandas will automatically installed as a dependency of `afd-measures`.
You can start with this Python snippet to analyse your own data (a CSV file in this example):
```python
import afd_measures
import pandas as pd

my_data = pd.read_csv("my_amazing_table.csv")
print(afd_measures.mu_plus(my_data, lhs="X", rhs="Y"))
```

## Installation (experiments)

To revisit the experiments that we did, clone this repository and install all requirements with [Poetry](https://python-poetry.org) (preferred) or [Conda](https://conda.io).

### Poetry

Install the requirements using poetry. Use the extra flag "experiments" to install all additional requirements for the experiments to work. This includes (amongst others) [Jupyter Lab](https://jupyter.org/).
```sh
$ poetry install -E experiments
$ jupyter lab
```

### Conda

Create a new environment from the `conda_environment.yaml` file, activate it and run Jupyter lab to investigate the code.

```sh
$ conda create -f conda_environment.yaml
$ jupyter lab
```

## Dataset References

In addition to this repository, we made our benchmark also available on Zenodo: [find it here](https://www.zenodo.org/record/8098909)

* `adult.csv`: Dua, D. and Graff, C. (2019). [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml). Irvine, CA: University of California, School of Information and Computer Science. 
* `claims.csv`: TSA Claims Data 2002 to 2006, [published by the U.S. Department of Homeland Security](https://www.dhs.gov/tsa-claims-data).
* `dblp10k.csv`: Frequency-aware Similarity Measures. Lange, Dustin; Naumann, Felix (2011). 243–248. [Made available as DBLP Dataset 2](https://hpi.de/naumann/projects/repeatability/datasets/dblp-dataset.html).
* `hospital.csv`: Hospital dataset used in Johann Birnick, Thomas Bläsius, Tobias Friedrich, Felix Naumann, Thorsten Papenbrock, and Martin Schirneck. 2020. Hitting set enumeration with partial information for unique column combination discovery. Proc. VLDB Endow. 13, 12 (August 2020), 2270–2283. https://doi.org/10.14778/3407790.3407824). [Made available as part the dataset collection to that paper.](https://owncloud.hpi.de/s/j6Z0yvXC0qhtGCk/download)
* `t_biocase_...` files: t\_bioc\_... files used in Johann Birnick, Thomas Bläsius, Tobias Friedrich, Felix Naumann, Thorsten Papenbrock, and Martin Schirneck. 2020. Hitting set enumeration with partial information for unique column combination discovery. Proc. VLDB Endow. 13, 12 (August 2020), 2270–2283. https://doi.org/10.14778/3407790.3407824). [Made available as part the dataset collection to that paper.](https://owncloud.hpi.de/s/j6Z0yvXC0qhtGCk/download)
* `tax.csv`: Tax dataset used in Johann Birnick, Thomas Bläsius, Tobias Friedrich, Felix Naumann, Thorsten Papenbrock, and Martin Schirneck. 2020. Hitting set enumeration with partial information for unique column combination discovery. Proc. VLDB Endow. 13, 12 (August 2020), 2270–2283. https://doi.org/10.14778/3407790.3407824). [Made available as part the dataset collection to that paper.](https://owncloud.hpi.de/s/j6Z0yvXC0qhtGCk/download)
