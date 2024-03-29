% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/wrapper.R
\name{check_layers}
\alias{check_layers}
\title{Function to check the basic shape of data between Layers}
\usage{
check_layers(layers, evaluator, dim_dataset, dim_target, batch_size)
}
\arguments{
\item{layers}{A list of descriptors}

\item{batch_size}{The batch_size to be used in the CNN or MLP}

\item{dataset}{A matrix with data, one example per row.}

\item{target}{A matrix with data targets, one example per row.}
}
\value{
TRUE if the list input and output dimensions match. Otherwise returns
FALSE.
}
\description{
This function checks if a list of layer descriptors is correctly dimensioned.
}
\examples{
dataset <- array(1:784000, c(1000, 1, 28, 28));
targets <- binarize(as.integer((rnorm(1000, 0, 1)  \%\% 1) * 10));
descriptor <- list(
c('type' = "CONV", 'n_channels' = 1, 'n_filters' = 4, 'filter_size' = 5, 'scale' = 0.1, 'border_mode' = 'same'),
c('type' = "POOL", 'n_channels' = 4, 'scale' = 0.1, 'win_size' = 3, 'stride' = 2),
c('type' = "RELU", 'n_channels' = 4),
c('type' = "CONV", 'n_channels' = 4, 'n_filters' = 16, 'filter_size' = 5, 'scale' = 0.1, 'border_mode' = 'same'),
c('type' = "POOL", 'n_channels' = 16, 'scale' = 0.1, 'win_size' = 3, 'stride' = 2),
c('type' = "RELU", 'n_channels' = 16),
c('type' = "FLAT", 'n_channels' = 16),
c('type' = "LINE", 'n_visible' = 784, 'n_hidden' = 64, 'scale' = 0.1),
c('type' = "RELV"),
c('type' = "LINE", 'n_visible' = 64, 'n_hidden' = 10, 'scale' = 0.1),
c('type' = "SOFT", 'n_inputs' = 10)
);
evaluator <- c('type' = "XENT"); # Cross-Entropy evaluator
is_valid <- check_layers (descriptor, evaluator, dim(training_x), dim(training_y), batch_size = 10);
}
\keyword{MLP}

