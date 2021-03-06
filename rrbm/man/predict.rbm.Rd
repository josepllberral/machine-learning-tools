% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/wrapper.R
\name{predict.rbm}
\alias{predict.rbm}
\title{Predicting using a Restricted Boltzmann Machine Function}
\usage{
\method{predict}{rbm}(rbm, newdata)
}
\arguments{
\item{rbm}{A trained RBM using train.rbm() function.}

\item{newdata}{A matrix with data, one example per row.}
}
\description{
Function to Predict data from a RBM. Returns two matrices: activation
and reconstruction.
}
\examples{
test_X <- t(array(c(1, 1, 0, 0, 0, 0,
                    0, 0, 0, 1, 1, 0), c(6,2)));
res <- predict(rbm1, test_X);

# The MNIST example
data(mnist)

training.num <- data.matrix(mnist$train$x)/255;
rbm_mnist <- train.rbm(n_hidden = 30, dataset = training.num,
                       learning_rate = 1e-3, training_epochs = 10,
                       batch_size = 10, momentum = 0.5);

passed_data <- predict(rbm_mnist, training.num);
}
\keyword{RBM}
