# prem4derg

This is a python module and some example notebooks to allow us (the Deep Earth Research Group at Leeds) to play around with
PREM, a famous 1D model of the Earth's interior published by Dziewonski and Anderson in 1981.

## Tasks that need doing
* Make graphs etc. in the density notebook nice.
* Fill in the 'blurb' around the graphs.
* Tabulate the parameters for Vp, Vs, Qkappa and Qmu.
* Add velocity functions to the PREM module that take r (scalar or an array) and a period (optional, default 1 s) and return the velocity.
* Dump out a obspy taup model from the velocities and use this to plot travel time curves
* Decorate with a subset of the travel time data used to build PREM.
* Create a normal modes notebook. Dump out models to mineos (https://geodynamics.org/cig/software/mineos/) and calculate some mode data used to fit PREM. Show comparison.
  

## References

Dziewonski, A. M. and Anderson, D. L. (1981) Preliminary reference Earth model *Physics of the Earth and 
Planetary Interiors* vol.25 pp.297—356. 
