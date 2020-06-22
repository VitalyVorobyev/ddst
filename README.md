# `X(cc ubar dbar) -> DD*` toy model study

A model for `J^P = 1^+` isoscalar resonance decaying to the finals states:

* `D0 D0 pi+`
* `D0 D+ pi0`
* `D0 D+ gamma`

is considered. T-matrix-based approach with two coupled elastic channels (`D0 D*+` and `D+ D*0`) is implemented.  Inelastic `DD` S-wave and P-wave amplitudes are considered, too.

## Dependences

* [`numpy`](numpy.org)
* [`matplotlib`](matplotlib.org)
* [`scipy`](scipy.org)

## [lib](lib) directory

Contains three-body decay kinematic toos, model implementation, model configuration, resolution convolution tools, and plotting tools.

* [params.py](./lib/params.py) keeps constants and configuration of the model
* [dalitzphsp.py](./lib/dalitzphsp.py) class `DalitzPhsp`. Common tools for a three-body decay kinematic. Decay model classes `DnDnPip`, `DnDpPin`, and `DnDpGam` inherit from the `DalitzPhsp` class.
* [lineshape.py](./lib/lineshape.py) class `TMtx` for the T-matrix amplitude. `RelativisticBreitWigner` lineshape
* [dndnpip.py](./lib/dndnpip.py): class `DnDnPip` describing the D0 D0 pi+ final state
* [dndppin.py](./lib/dndnpip.py): class `DnDpPin` describing the D0 D+ pi0 final state
* [dndpgam.py](./lib/dndngam.py): class `DnDpGam` describing the D0 D+ gamma final state
* [resolution.py](./lib/resolution.py) contains resolution convolution tools for energy, DD kinetic energy, and D0 pi+ mass
* [plot.py](./lib/plot.py) contains auxiliary plotting tools (energy spectrum, Dalitz plot distribution, projections of the Dalitz plot)

## [root](.) directory

Contains drivers for various actions with models from [lib](lib):

* [contours.py](./contours.py) produces filled contour plot for the decay probability dencity on the (`E`, `T(DD)`) plane
* [dp.py](./dp.py) produces Dalitz plot distribution and it's projections for a given `E`
* [espec.py](./espec.py) produces energy spectrum via grid integrals over the Dalitz plot phase space (w/o resolution convolution)
* [observables.py](./observables.py) produces **summary plot** of all observables for the current model configuration
