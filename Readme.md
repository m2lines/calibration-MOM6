Repository supporting paper [Calibration of a neural network ocean closure for improved mean state and variability](https://arxiv.org/abs/2604.06398) by Pavel Perezhogin, Alistair Adcroft and Laure Zanna.

* Notebooks with figures plotted in the paper:
[paper_figures.ipynb](https://github.com/m2lines/calibration-MOM6/blob/publication/notebooks/paper-figures.ipynb)
* Calibration script: [driver.py](https://github.com/m2lines/calibration-MOM6/blob/publication/calibration_driver/driver.py)
* Parameters used to run calibration scripts: [configs](https://github.com/m2lines/calibration-MOM6/blob/publication/configs)
* Implementation of the equivariant neural network in Torch: [ann_tools.py](https://github.com/m2lines/calibration-MOM6/blob/publication/scripts/ann_tools.py#L139-L185)
* Weights of the offline-trained and calibrated neural networks [eANN_weights](https://github.com/m2lines/calibration-MOM6/tree/publication/eANN_weights)
* Simulation and all supporting data can be found on [Zenodo](https://doi.org/10.5281/zenodo.20261601)
* Parameterization is implemented in the main branch of [GFDL MOM6](https://github.com/NOAA-GFDL/MOM6.git)
