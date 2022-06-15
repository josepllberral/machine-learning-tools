# Machine Learning Tools

Machine Learning tools and functions in R, C and Python + Tensorflow, for educational and academic purposes.

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
