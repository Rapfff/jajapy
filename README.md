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

<div align="center">
	
| Markov Model   |      Learning Algorithm(s) |
|-------|:-------------:|
| HMM    | Baum-Welch for HMMs  ([ref](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)) |
| MC     | Baum-Welch for MCs <br /> Alergia ([ref](https://www.researchgate.net/publication/2543721_Learning_Stochastic_Regular_Grammars_by_Means_of_a_State_Merging_Method/stats)) |
| MDP    | Baum-Welch for MDPs ([ref](https://arxiv.org/abs/2110.03014))<br /> Active Baum-Welch ([ref](https://arxiv.org/abs/2110.03014))<br /> IOAlergia ([ref](https://link.springer.com/content/pdf/10.1007/s10994-016-5565-9.pdf))|
| CTMC   | Baum-Welch for CTMCs<br /> MM for asynchronous composition of CTMCs|
| GoHMM  | Baum-Welch for GoHMMs ([ref](http://www.inass.org/2020/2020022920.pdf)) |
| MGoHMM | Baum-Welch for MGoHMMs |

</div>

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
