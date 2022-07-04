# Machine Learning Tools

Machine Learning tools and functions in R, C and Python + Tensorflow, for educational and academic purposes.

Here you can find (C)RBMs, and configurable MLP (including Convolutional layers) implemented in R.

### R Files

#### Feed-Forward MultiLayer Perceptron Networks

An implementation of FFANNs in R, with single hidden layer and multiple hidden layers.

* **ffann.R**: Single hidden layer network for classification with softmax output.
* **mlp.R**: Quick Multi-Layer Perceptron FFANN with softmax output.
* **cnn.R**: Configurable MultiLayer Perceptron in R, including Convolutional, Pooling, ReLU, Linear, Softmax, Sigmoid, TanH and Direct layers.

#### Restricted Boltzmann Machines

Here you can find an implementation of RBMs and CRBMs in R. The code comes with an example for the MNIST dataset, and an example for the MOTION dataset (you can find an RDS version for the motion dataset in the "datasets" folder).

* **rbm.R**: RBMs in R.
* **crbm.R**: Conditional RBMs in R.
* **crbm_series.R**: Extension of crbm.R, allowing training from sets of series instead a single one in parallel.

#### Kalman Filters

An implementation of Kalman Filters in R. The code comes with an example for time series.

* **kalman_filter.R**: Version in R.

### Packages for R

You can find the (C)RBMs, MLPs and CNNs ready to install and use in R, in the **packages** folder.

#### Package sources

In the **rcnn** and **rrbm** folders you can find the sources for the R packages, with the kernel functions in C.

#### Build and Test tools

Also, the **test** folder has the R code for building and testing the R packages.

### Datasets

In the **datasets** folder, you can find the example datasets for the here found tools:

* **motion.rds**: version of the motion.mat dataset in RDS format. This dataset is a fragment of Eugene Hsu [Styles of Human Motion](http://people.csail.mit.edu/ehsu/work/sig05stf/), taken by Graham Taylor for his works [here](http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/motion.mat) .

* **mnist.rds**: The MNIST digit recognition dataset is from Yann LeCun's [MNIST Database](http://yann.lecun.com/exdb/mnist/) in RDS format.