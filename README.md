# DecEnSys - urbs

urbs is a [linear programming](https://en.wikipedia.org/wiki/Linear_programming) optimisation model for capacity expansion planning and unit commitment for distributed energy systems. Its name, latin for city, stems from its origin as a model for optimisation for urban energy systems. Since then, it has been adapted to multiple scales from neighbourhoods to continents.

[![Documentation Status](https://urbs.readthedocs.io/en/decensys/)](https://readthedocs.org/projects/urbs/badge/?version=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.242029.svg)](https://doi.org/10.5281/zenodo.242029)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/tum-ens/urbs?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

## DecEnSys project

This branch features the further development of urbs during the project [DecEnSys](https://www.ens.ei.tum.de/en/research/projects/current-projects/decensys/). It includes three main running modes: regional and time decomposition and stochastic dual dynamic programming (SDDP).

## Contributers

The code basis is, of course, urbs. Next to the urbs contributers the work in the project was conducted by Magdalena Stüber, Paul Stursberg, Leonhard Odersky, Okan Akca and Christoph Hahn.

## Features

  * urbs is a linear programming model for multi-commodity energy systems with a focus on optimal storage sizing and use.
  * It finds the minimum cost energy system to satisfy given demand timeseries for possibly multiple commodities (e.g. electricity).
  * By default, operates on hourly-spaced timesteps (configurable).
  * Thanks to [Pandas](https://pandas.pydata.org), complex data analysis is easy.
  * The model itself is quite small thanks to relying on package [Pyomo](http://www.pyomo.org/).
  * The small codebase includes reporting and plotting functionality.
  * Stochastic evaluations of energy systems

## Screenshots

<a href="doc/img/plot.png"><img src="doc/img/plot.png" alt="Timeseries plot of 8 days of electricity generation in vertex 'North' in scenario_all_together in hourly resolution: Hydro and biomass provide flat base load of about 50% to cover the daily fluctuating load, while large share of wind and small part photovoltaic generation cover the rest, supported by a day-night storage." style="width:400px"></a>

<a href="doc/img/comparison.png"><img src="doc/img/comparison.png" alt="Bar chart of cumulated annual electricity generation costs for all 5 scenarios defined in runme.py." style="width:400px"></a>

## Installation

### Windows, Linux, Mac OS

The easiest way to get all required packages under all distributions it to use Anaconda or Miniconda. If you don't want to use it or already have an existing Python (version 3.7 **recommended**) installation, you can also download the required packages by yourself.

#### Anaconda (recommended)

  1. **[Anaconda (Python 3.7)](http://continuum.io/downloads)**. Choose the 64-bit installer if possible.  
     During the installation procedure, keep both checkboxes "modify PATH" and "register Python" selected!
  2. **Solver, additional packages**: Use the provided environment file and install it by: `conda env create -f urbs-decensys.yml`

 
## Get started

Once installation is complete, finally [install git (for version control)](http://git-scm.com/). **Remark:** at step "Adjusting your PATH environment", select "Run Git from the Windows Command Prompt".

Then, in a directory of your choice, clone this repository and execute the runme script by executing the following on the command prompt (Windows) or Terminal (Linux): 

    git clone https://github.com/tum-ens/urbs.git
    cd urbs
    conda activate decensys
    python runme.py

Some minutes later, the subfolder `result` should contain plots and summary spreadsheets for multiple optimised energy supply scenarios, whose definitions are contained in the run script (watch out for `def scenario` lines).

For the decensys branch, more steps need to be taken like setting respective parameters in `runme.py`.

## Next steps

  1. Head over to the tutorial at http://urbs.readthedocs.io, which goes through runme.py step by step. 
  2. Read the source code of `runme.py` and `comp.py`. 
  3. Quickly scan through `models`, read docstrings.
  4. Try adding/modifying scenarios in `runme.py` and see their effect on results.
  5. Fire up IPython (`ipython3`) and run the scripts from there using the run command: `run runme`. Then use `whos` and inspect the workspace afterwards (`whos`). See what you can do (analyses, plotting) with the DataFrames. Take the `urbs.get_constants`, `urbs.get_timeseries` and `urbs.plot` functions as inspriation and the [Pandas docs](http://pandas.pydata.org/pandas-docs/stable/) as reference.
  
## Further reading

  - If you do not know anything about the command line, read [Command Line Crash Course](https://learnpythonthehardway.org/book/appendixa.html). Python programs are scripts that are executed from the command line, similar to MATLAB scripts that are executed from the MATLAB command prompt.
  - If you do not know Python, try one of the following ressources:
    * The official [Python Tutorial](https://docs.python.org/3/tutorial/index.html) walks you through the language's basic features.
    * [Learn Python the Hard Way](https://learnpythonthehardway.org/book/preface.html). It is meant for programming beginners.
  - The book [Python for Data Analysis](http://shop.oreilly.com/product/0636920023784.do) best summarises the capabilities of the packages installed here. It starts with IPython, then adds NumPy, slowly fades to pandas and then shows first basic, then advanced data conversion and analysis recipes. Visualisation with matplotlib is given its own chapter, both with and without pandas.
  - For a huge buffet of appetizers showing the capabilities of Python for scientific computing, I recommend browsing this [gallery of interesting IPython Notebooks](https://github.com/ipython/ipython/wiki/A-gallery-of-interesting-IPython-Notebooks).
  
## Example uses

  - A short term case study for Germany has been conducted with the SDDP approach for the [EMP-E conference 2019](http://www.energymodellingplatform.eu/home-emp-e-2019.html) and is available on [zenodo](https://doi.org/10.5281/zenodo.3463157).


## Copyright

Copyright (C) 2014-2019  TUM ENS

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
