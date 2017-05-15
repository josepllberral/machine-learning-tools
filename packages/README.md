# Machine Learning Tools

## R Packages (Using GSL)

**Note: Package RCNN has a bug and it is under revision!**

Here you can find the (C)RBMs, MLPs and CNNs ready to install and use.

* Package **rrbm** : contains RBM and CRBM implementations, with training, predict and forecasting functions. Also MNIST and Motion (fragment) datasets are included ("mnist" and "motionfrag").

* Package **rcnn** : contains a MLP modular implementation, with different layer definitions like *Linear*, *Convolutional*, *Pooling*, *Flattening*, *Linear Rectifier*, *Softmax*, etc. Also contains training and predict functions. Also MNIST datasets is included ("mnist").

As the packaged tools use a GSL implementation as kernel (written in C, but interfaced to R), installing the GSL development libraries is required:

> apt install libgsl-dev

Once "gsl" libraries are installed, just do

> install.packages("rrbm_0.0.X.tar.gz", repos = NULL, type="source")

> install.packages("rcnn_0.0.X.tar.gz", repos = NULL, type="source")

In case you don't want to install GSL or use C code, you can just download the **pure-R** implementation from the *.R files in this same repository. They do exactly the same than the GSL-core functions, but (often) slower.

