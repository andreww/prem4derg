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

    wget http://geoweb.cse.ucdavis.edu/cig/software/mineos/mineos-1.0.2.tgz
    mkdir cig
    tar -xzvf mineos-1.0.2.tgz
    cd mineos-1.0.2
    ./configure --prefix=/Users/andreww/Code/cig

Edit the Makefile to add: `-Wno-error -Wno-return-type` to the `CFLAGS` because the C contains errors.
Add `-fallow-argument-mismatch` to FFLAGS because the Fortran contains errors.

    make 
    make install

## Examples of use

Example Jupyter notebooks can be accessed and explored by running `jupyter notebook`. Four examples are
currently provided:

* A [density example](./PREM_density_example.ipynb), showing how a model can be defined and used to calculate mass, moment of inertia, gravity and pressure.
* A [velocity example](./PREM_velocity_example.ipynb), showing how a model can be defined and used to calculate seismic velocities as a function of depth and period.
* A [travel time example](./PREM_travel_times_example.ipynb), showing how an obspy taupy model can be created and used to compute travel time curves.
* A [normal modes example](./PREM_normal_modes_example.ipynb), showing how Mineos can be used to compute normal mode frequencies.

These examples are only starting point. For example, the code could be used to fit new models. 

## Development and support

Prem4derg is new software, bugs and rough edges may abound. Users who are interested in 
changing the code are encouraged to [create a fork on GitHub and submit 
changes via pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests).
Problems can be reported via [issues](https://github.com/andreww/prem4derg/issues), which also list areas
where further development may be useful.
  

## References

Dziewonski, A. M. and Anderson, D. L. (1981) Preliminary reference Earth model *Physics of the Earth and 
Planetary Interiors* vol.25 pp.297—356. 
