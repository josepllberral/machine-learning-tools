% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mnist-data.R
\docType{data}
\name{mnist}
\alias{mnist}
\title{MNIST dataset}
\format{List containing
\itemize{
  \item train: list of 3 elements:
  \itemize{
    \item n : number of elements (actually 60000).
    \item x : integer matrix of 60000 x 784. Rows are flattened 28 x 28 digit images.
    \item y : integer vector of 60000. Labels for each row in x.
  }
  \item test: list of 3 elements:
  \itemize{
    \item n : number of elements (actually 10000).
    \item x : integer matrix of 10000 x 784. Rows are flattened 28 x 28 digit images.
    \item y : integer vector of 10000. Labels for each row in x.
  }
}}
\source{
\href{http://yann.lecun.com/exdb/mnist/}{The MNIST Database}
}
\usage{
data(mnist)
}
\description{
Data containing handwritten digits, from Yann LeCun, Corina Cortes, and
Christopher J.C. Burges.
}
\examples{
data(mnist)
\donttest{head(mnist$train$x) / 255}
train_y <- mnist$train$y;
\donttest{table(train_y)}
test_y <- mnist$test$y;
\donttest{table(test_y)}
}
\references{
LeCun et al. (2010)
(\href{http://yann.lecun.com/exdb/mnist/}{The MNIST Database})
}
\keyword{datasets}

