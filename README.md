# Machine Learning Tools

Machine Learning tools and functions in R, for educational and academic purposes.

## Tools

### Conditional Restricted Boltzmann Machines

Here you can find an implementation of CRBMs in R, based on the approach of Graham Taylor's CRBMs in Python and Theano. Despite that, here's implementation is in plain code (no optimization or vectorization). The code comes with an example for the MOTION dataset (you can find an RDS version for the motion dataset in the datasets folder).

* crbm.R: Version in R, with some improvements.
* crbm_gt.R: Version directly translated from G.Taylor's, with indications to compare against the original version at https://gist.github.com/gwtaylor/2505670

### Restricted Boltzmann Machines

Here you can find an implementation of RBMs in R. The code comes with an example for the MNIST dataset.

* rbm.R: Version in R

### Kalman Filters

Here you can find an implementation of Kalman Filters in R. The code comes with an example for time series.

* kalman_filter.R: Version in R

## Example Notebooks

Notebooks in _jupyter_ for the CRBMs, RBMs and Kalman Filter in R.

## Datasets

Example datasets for the here found tools:

* motion.rds: version of the motion.mat dataset (Eugene Hsu http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/motion.mat) in RDS format.
