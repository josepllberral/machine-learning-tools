# Machine Learning Tools

Machine Learning tools and functions in R, for educational and academic purposes.

## R Packages (Using GSL)

You can find the (C)RBMs, MLPs and CNNs ready to install and use, in the **packages** folder.

* Package **rrbm** : contains RBM and CRBM implementations, with training, predict and forecasting functions. Also MNIST and Motion (fragment) datasets are included ("mnist" and "motionfrag").

* Package **rcnn** : contains a MLP modular implementation, with different layer definitions like *Linear*, *Convolutional*, *Pooling*, *Flattening*, *Linear Rectifier*, *Softmax*, etc. Also contains training and predict functions. Also MNIST datasets is included ("mnist").

As the packaged tools use a GSL implementation as kernel (written in C, but interfaced to R), installing the GSL development libraries is required:

> apt-get install libgsl-dev

Once "gsl" libraries are installed, just do

> install.packages("rrbm_0.0.X.tar.gz", repos = NULL, type="source")

> install.packages("rcnn_0.0.X.tar.gz", repos = NULL, type="source")

Further, you can install "OpenBLAS" (optional) to let GSL use multi-processor computation:

> apt-get install libopenblas-base

In case you don't want to install GSL or use C code, you can just download the **pure-R** implementation from the *.R files in this same repository. They do exactly the same than the GSL-core functions, but (often) slower.


## Tools Description

Here you can find (C)RBMs, and configurable MLP (including Convolutional layers) implemented in R.

### Feed-Forward and MultiLayer Perceptrons

An implementation of FFANNs in R, with single hidden layer and multiple hidden layers.

* **ffann.R**: Single hidden layer network for classification with softmax output.
* **mlp.R**: Quick Multi-Layer Perceptron FFANN with softmax output.

Also a configurable version of MLPs in R, including Convolutional, Pooling, ReLU, Linear, Softmax/Sigmoid/TanH/Direct layers.

* **cnn.R**: Version in R.

### Restricted Boltzmann Machines

Here you can find an implementation of RBMs in R. The code comes with an example for the MNIST dataset.

* **rbm.R**: Version in R.

Also an implementation of Conditional RBMs in R. The code comes with an example for the MOTION dataset (you can find an RDS version for the motion dataset in the "datasets" folder).

* **crbm.R**: Version in R.
* **crbm_series.R**: Extension of crbm.R, allowing training from sets of series instead a single one in parallel.

### Kalman Filters

An implementation of Kalman Filters in R (because not everything is going to be neural networks!). The code comes with an example for time series.

* **kalman_filter.R**: Version in R.

## Example Notebooks

In the "notebooks" folder you can find the R code, detailed and executed. Notebooks are in _jupyter_ format.


## Datasets

Example datasets for the here found tools:

* **motion.rds**: version of the motion.mat dataset in RDS format. This dataset is a fragment of Eugene Hsu [Styles of Human Motion](http://people.csail.mit.edu/ehsu/work/sig05stf/), taken by Graham Taylor for his works [here](http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/motion.mat) .

* **mnist.rds**: The MNIST digit recognition dataset is from Yann LeCun's [MNIST Database](http://yann.lecun.com/exdb/mnist/) in RDS format.

## Acknowledgements

The FFANN approach takes inspiration from Peng Zhao ["R for Deep Learning"](http://www.parallelr.com/r-deep-neural-network-from-scratch).

The CNN approach takes inspiration from Lars Maaloee's [Tutorial on CNNs](https://github.com/davidbp/day2-Conv), also from Yann LeCun's [deeplearning.net](http://deeplearning.net/tutorial/lenet.html).

The CRBMs approach is based on Graham Taylor's [CRBMs in Python and Theano](https://gist.github.com/gwtaylor/2505670).

