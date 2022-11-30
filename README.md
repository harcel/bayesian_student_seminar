# The how and why of statistically modeling student results

Seminar at the UvA Data Science Center on Nov 25, 2022

This is a demo, in which bayesian inference is used to model student ability and course difficulty on observed student grades.

It begins by demoing the idea of bayesian generative modeling with simple examples. 
After that, a model for student grades is presented and tested in an idealized situation.
Real data of UvA students is then modeled to demonstrate possible use cases.

The file `utils.py` contains many utility functions to keep the notebook clean (but hopefully still self-explanatory).
Note that running the whole notebook is going to crash on you at some point, as the student data is not in the repository for obvious reasons. In the demo, only aggregated data is shown, or very small pseudonymized subsets as examples.

To install a conda environment that can run the code:

`conda env create -f environment.yml`

The notebook uses Rise.js to make it a slideshow, which makes some cells look a bit odd in notebook mode.


Marcel Haas, Nov 2022
m.r.haas@uva.nl
