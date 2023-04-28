<div align="center">
<h1>
<img src="https://raw.githubusercontent.com/Rapfff/jajapy/main/logo.png" width="300">
</h1><br>

[![Pypi](https://img.shields.io/pypi/v/jajapy)](https://pypi.org/project/jajapy/)
[![Python 3.6](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/release/python-360/)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/aalpy)
[![Documentation Status](https://readthedocs.org/projects/jajapy/badge/?version=latest)](https://jajapy.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/Rapfff/jajapy)](https://en.wikipedia.org/wiki/MIT_License)
</div>


## Introduction
`jajapy` is a python library implementing the **Baum-Welch** algorithm on various kinds of Markov models.
`jajapy` generates models which are compatible with the Stormpy model checker. Thus, `jajapy`can be use as a learning extension to the Storm model checker.


## Main features
`jajapy` provides:

<div align="center">
	
| Markov Model   |      Learning Algorithm(s) |
|-------|:-------------:|
| MC     | Baum-Welch for MCs <br /> Alergia ([ref](https://www.researchgate.net/publication/2543721_Learning_Stochastic_Regular_Grammars_by_Means_of_a_State_Merging_Method/stats)) |
| MDP    | Baum-Welch for MDPs ([ref](https://arxiv.org/abs/2110.03014))<br /> Active Baum-Welch ([ref](https://arxiv.org/abs/2110.03014))<br /> IOAlergia ([ref](https://link.springer.com/content/pdf/10.1007/s10994-016-5565-9.pdf))|
| CTMC   | Baum-Welch for CTMCs <br /> Baum-Welch for synchronous compositions of CTMCs|
| PCTMC  | Baum-Welch for PCTMCs ([ref](https://arxiv.org/abs/2302.08588))|
| HMM    | Baum-Welch for HMMs  ([ref](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)) |
| GoHMM  | Baum-Welch for GoHMMs ([ref](http://www.inass.org/2020/2020022920.pdf)) |

</div>

`jajapy` is compatible with [Prism](http://www.prismmodelchecker.org/) and [Storm](https://www.stormchecker.org/).

## Installation
``pip install jajapy``

## Requirements
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [alive-progress](https://github.com/rsalmei/alive-progress) 
- [sympy](https://www.sympy.org/en/index.html)
- [stormpy](https://github.com/moves-rwth/stormpy) (recommended: if stormpy is not installed, `jajapy` will generate models in jajapy format).

## Documentation
Available on [readthedoc](https://jajapy.readthedocs.io/en/latest/?)

## About the author
[My website](https://rapfff.github.io/)