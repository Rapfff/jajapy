<div align="center">
<h1>
<img src="logo.png" width="300">
</h1><br>

[![Pypi](https://img.shields.io/pypi/v/jajapy)](https://pypi.org/project/jajapy/)
[![Python 3.6](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/release/python-360/)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/aalpy)
[![Documentation Status](https://readthedocs.org/projects/jajapy/badge/?version=latest)](https://jajapy.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/Rapfff/jajapy)](https://en.wikipedia.org/wiki/MIT_License)
</div>


## Introduction
`jajapy` is a python library implementing the **Baum-Welch** algorithm on various kinds of Markov models.

Please cite this repository if you use this library.

## Main features
`jajapy` provides:
- BW algorithm for Hidden Markov Models [reference](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)
- BW algorithm for Markov Chains
- BW algorithm for Gaussian Observation Hidden Markov Models [reference](http://www.inass.org/2020/2020022920.pdf)
- BW algorithm for Markov Decision Processes [reference](https://arxiv.org/abs/2110.03014)
- Active BW algorithm for Markov Decision Processes [reference](https://arxiv.org/abs/2110.03014)
- BW algorithm for CTMC
- BW algorithm for asynchronous parallel composition of CTMCs

Additionally, it provides other learning algorithms:
- Alergia, for Markov Chains [reference](https://www.researchgate.net/publication/2543721_Learning_Stochastic_Regular_Grammars_by_Means_of_a_State_Merging_Method/stats)
- IOAlergia, for Markov Decision Processes [reference](https://link.springer.com/content/pdf/10.1007/s10994-016-5565-9.pdf)

## Installation
``pip install jajapy``

## Requirements
- numpy
- scipy

## Documentation
Available on [readthedoc](https://jajapy.readthedocs.io/en/latest/?)

## TO DO
- Add examples in the documentation
- link with stormpy, prism
- errors management
