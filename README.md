# prem4derg

This is a python module and some example notebooks to allow us (the Deep Earth Research Group at Leeds) to play around with
PREM, a famous 1D model of the Earth's interior published by Dziewonski and Anderson in 1981.

## Installation

This module relies on python (version 3), Numpy and Scipy. Examples are distributed as Jupyter notebooks, which need Jupyter
and Matplotlib to run. Calculation of travel time curves (in the third example notebook) needs Obspy. Installation and managment
of all these dependencies is most easily done in a conda environment. This can be done using the command:

    conda create -n p4d -c conda-forge python=3.7 obspy scipy numpy matplotlib jupyter
   
This creates a new environment called `p4d`. Once an environment is created, this module can be downloaded by running:

    git clone https://github.com/andreww/prem4derg.git
    
This creates a directory called `prem4derg`. Experianced users of git who want to contribute to the code may want to fork this
repository in github and clone from their fork.

Two final steps will need to be done each time you want to use the code. These are to change directory into the `prem4derg` directory,
and activate the `p4d` environemnt by running `source activate p4d`.

### Mineos

In order to compute normal mode frequencies, we use [mineos](https://geodynamics.org/cig/software/mineos/). I 
installed this as follows (working outside the prem4derg directory):

    wget https://geodynamics.org/cig/software/mineos/mineos-1.0.2.tgz
    mkdir cig
    tar -xzvf mineos-1.0.2.tgz
    cd mineos-1.0.2
    ./configure --prefix=/Users/andreww/Code/cig

Edit the Makefile to add: `-Wno-error -Wno-return-type` to the `CFLAGS` because the C contains errors.

    make 
    make install



## Examples of use

Example Jupyter notebooks can be accessed and explored by running `jupyter notebook`. 

## Tasks that need doing
* Make graphs etc. in the density notebook nice.
* Fill in the 'blurb' around the graphs.
* Dump out a obspy taup model from the velocities and use this to plot travel time curves
* Decorate with a subset of the travel time data used to build PREM.
* Create a normal modes notebook. Dump out models to mineos (https://geodynamics.org/cig/software/mineos/) and calculate some mode data used to fit PREM. Show comparison.
  

## References

Dziewonski, A. M. and Anderson, D. L. (1981) Preliminary reference Earth model *Physics of the Earth and 
Planetary Interiors* vol.25 pp.297—356. 
