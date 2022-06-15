# Machine Learning Tools

Machine Learning tools and functions in R, C and Python + Tensorflow, for educational and academic purposes.

## R Tools

In the **R tools** folder you can find (C)RBMs, and configurable MLP (including Convolutional layers) implemented in R.

### Tools and Layers

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

* Package **rrbm** : contains RBM and CRBM implementations, with training, predict and forecasting functions. Also MNIST and Motion (fragment) datasets are included ("mnist" and "motionfrag").

* Package **rcnn** : contains a MLP modular implementation, with different layer definitions like *Linear*, *Convolutional*, *Pooling*, *Flattening*, *Linear Rectifier*, *Softmax*, etc. Also contains training and predict functions. Also MNIST datasets is included ("mnist").

As the packaged tools use a GSL implementation as kernel (written in C, but interfaced to R), installing the GSL development libraries is required. Further, you can install "OpenBLAS" (optional) to let GSL use multi-processor computation:

> apt-get install libgsl-dev

> apt-get install libopenblas-base

Once "gsl" libraries are installed, just do in R:

> install.packages("rrbm_0.0.X.tar.gz", repos = NULL, type="source")

> install.packages("rcnn_0.0.X.tar.gz", repos = NULL, type="source")

In case you don't want to install GSL or use C code, you can just download the **pure-R** implementation from the \*.R files in this same repository. They do exactly the same than the GSL-core functions, but (often) slower.

#### Package sources

In the **rcnn** and **rrbm** folders you can find the sources for the R packages, with the kernel functions in C.

#### Build and Test tools

Also, the **test** folder has the R code for building and testing the R packages.

### Datasets

In the **datasets** folder, you can find the example datasets for the here found tools:

* **motion.rds**: version of the motion.mat dataset in RDS format. This dataset is a fragment of Eugene Hsu [Styles of Human Motion](http://people.csail.mit.edu/ehsu/work/sig05stf/), taken by Graham Taylor for his works [here](http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/motion.mat) .
* **mnist.rds**: The MNIST digit recognition dataset is from Yann LeCun's [MNIST Database](http://yann.lecun.com/exdb/mnist/) in RDS format.

## C Tools

In the  **C tools** folder you can find the (C)RBM and configurable MLPs (including Convolutional layers) implemented in C, for standalone use.

The C code is the equivalent for the C kernel in the R packages, but prepared for using it standalone, either as a library, application or included in any other code.

### Tools and Layers

#### Feed-Forward MultiLayer Perceptron Networks

An implementation of FFANNs in C, with single hidden layer and multiple hidden layers.

Main programs:
* **main.c**: Program entry point, with data loading
* **test.c**: Check the gradient for each layer

Network engines:
* **pipeline_cnn.c**: pipeline for interpreting 3d-image CNN networks (with Convolutional layers)
* **pipeline_mlp.c**: pipeline for interpreting flat-image MLP networks (without convolutional layers)

Layer definitions:
* **conv.c**: Convolutional Layer
* **dire.c**: Direct Layer
* **flat.c**: Flattening Layer
* **line.c**: Linear Layer
* **msel.c**: Mean Squared Error Evaluator
* **pool.c**: Pooling Layer
* **rbml.c**: Restricted Boltzmann Machine Layer
* **relu.c**: ReLU (matrix) Layer
* **relv.c**: ReLU (vector) Layer
* **sigm.c**: Sigmoid Layer
* **soft.c**: SoftMax Layer
* **tanh.c**: ArcTangent Layer
* **xent.c**: Cross-Entropy Evaluator

Auxiliar libraries:
* **grad_check.c**: utility to check the gradient for a layer
* **operations.c**: Matrix operations and auxiliar functions

### Datasets

In the **datasets** folder, you can find the example datasets for the here found tools:

* **MOTION** (fragment) binarized data files.
  * motion_bd.data
  * motion_dm.data
  * motion_ds.data
  * motion_sl.data
* **MNIST** binarized data files.
  * mnist_testx.data
  * mnist_testy.data
  * mnist_trainx.data
  * mnist_trainy.data

## TensorFlow Tools

You can find (C)RBMs implemented in Python + TensorFlow, in the **tensorflow** folder.

* **tf_rbm.py**: RBMs in TensorFlow
* **tf_crbm.py**: CRBMs in TensorFlow

In the inner **datasets** folder, you can find the example datasets for the here found tools:

* **motion.mat**: Matlab file by Graham Taylor ([original file](http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/motion.mat))


## Notebooks

You can find "notebooks" (in _jupyter_ format) explaining the R and TF code in detail, also examples of use for them, in the **notebooks** folder.

## Acknowledgements

The FFANN approach takes inspiration from Peng Zhao ["R for Deep Learning"](http://www.parallelr.com/r-deep-neural-network-from-scratch).

The CNN approach takes inspiration from Lars Maaloee's [Tutorial on CNNs](https://github.com/davidbp/day2-Conv), also from Yann LeCun's [deeplearning.net](http://deeplearning.net/tutorial/lenet.html).

The CRBMs approach is based on Graham Taylor's [CRBMs in Python and Theano](https://gist.github.com/gwtaylor/2505670).

