# Gawain

This is a simple python MHD simulation code. Currently only 2D simulations are supported. 

The numerical method is a MUSCL-Hancock scheme on an Eulerian grid with the minmod flux limiter.

## Getting Started

#### Prerequisites

The aim of this code was simplicity so the dependencies have been kept to a minimum. However, a few key external libraries are required and can be installed using 

```
 pip install -r requirements.txt
```

#### Installing

The package is set up using

```
python ./setup.py install
```

## Examples

A few example simulation scripts and jupyter notebooks can be found in the 'example_scripts' directory. The notebooks provide a detailed how-to on creating and running a simulation.

To run a simulation using a script, follow these steps:

1. Specify the parameters and the initial and boundary conditions in a`run_name.py` script. See the 'example_scripts' directory for some examples.
2. Run with  `python run_name.py` 
4. Output will be dumped into a folder  `output/<name>`. Where `<name>` is the simulation name specified in the run script.  