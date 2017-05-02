###############################################################################
# Wrapper functions for CNN library                                           #
###############################################################################


load_mnist <- function()
{
	load_image_file <- function(filename)
	{
		ret <- list();
		f <- file(filename, 'rb');
		readBin(f, 'integer', n = 1, size = 4, endian = 'big');
		ret$n <- readBin(f, 'integer', n = 1, size = 4, endian = 'big');
		nrow <- readBin(f, 'integer', n = 1, size = 4, endian = 'big');
		ncol <- readBin(f, 'integer', n = 1, size = 4, endian = 'big');
		x <- readBin(f, 'integer', n = ret$n * nrow * ncol, size = 1, signed = FALSE);
		ret$x <- matrix(x, ncol = nrow * ncol, byrow = TRUE);
		close(f);
		ret;
	}
	load_label_file <- function(filename)
	{
		f <- file(filename, 'rb');
		readBin(f, 'integer', n = 1, size = 4, endian = 'big');
		n <- readBin(f, 'integer', n = 1, size = 4, endian = 'big');
		y <- readBin(f, 'integer', n = n, size = 1, signed = FALSE);
		close(f);
		y;
	}
	train <- load_image_file('../rrbm/datasets/train-images.idx3-ubyte');
	test <- load_image_file('../rrbm/datasets/t10k-images.idx3-ubyte');

	train$y <- load_label_file('../rrbm/datasets/train-labels.idx1-ubyte');
	test$y <- load_label_file('../rrbm/datasets/t10k-labels.idx1-ubyte');

	list(train = train, test = test);
}

binarization <- function(vec)
{
	result <- array(0, c(length(vec),length(unique(vec))));
	for (i in 1:length(vec)) result[i,vec[i]] <- 1;
	result;
}

aux <- load_mnist();
img_size <- c(28,28);

train <- aux$train;
training_x <- array(train$x, c(nrow(train$x), 1, img_size)) / 255;
training_y <- binarization(train$y);

test <- aux$test;
testing_x <- array(test$x, c(nrow(test$x), 1, img_size)) / 255;
testing_y <- binarization(test$y);

dataset <- training_x[1:1000,];
targets <- training_y[1:1000,];

batch_size <- 10;
training_epochs <- 1;
learning_rate <- 1e-3;
momentum <- 0.8;
rand_seed <- 1234;

border_mode <- 2;
filter_size <- 5;

win_size <- 3;
stride <- 2;

layers <- list(
	c("CONV", 1, 4, filter_size, 0.1, border_mode, batch_size),
	c("POOL", 4, 0.1, batch_size, win_size, stride),
	c("RELU", 4, batch_size),
	c("CONV", 4, 16, filter_size, 0.1, border_mode, batch_size),
	c("POOL", 16, 0.1, batch_size, win_size, stride),
	c("RELU", 16, batch_size),
	c("FLAT", 16, batch_size),
	c("LINE", 784, 64, 0.1, batch_size),
	c("RELV", batch_size),
	c("LINE", 64, 10, 0.1, batch_size),
	c("SOFT", 10, batch_size)
);

dyn.load("cnn.so")

retval <- .Call("_C_CNN_train", as.array(dataset), as.matrix(targets), as.list(layers), as.integer(length(layers)), as.integer(batch_size), as.integer(training_epochs), as.double(learning_rate), as.double(momentum), as.integer(rand_seed));




#' Training a Convolutional Neural Network Function
train.cnn <- function (dataset, targets, layers,  batch_size = 10,
			training_epochs = 10, learning_rate = 1e-3,
			momentum = 0.8, rand_seed = 1234)
{
	if ("integer" %in% class(dataset[1,1]))
	{
		message("Input Array is not Numeric");
		return(NULL);
	}

	#if (!is.loaded("bscnn")) library.dynam("bscnn", package=c("bscnn"), lib.loc=.libPaths());

	retval <- .Call("_C_CNN_train", as.array(dataset), as.matrix(targets),
		as.list(layers), as.integer(length(layers)), as.integer(batch_size),
		as.integer(training_epochs), as.double(learning_rate), as.double(momentum),
		as.integer(rand_seed), PACKAGE = "rcnn");

	class(retval) <- c("cnn", class(retval));

	retval;
}

###################################################################

#' Predicting using a Restricted Boltzmann Machine Function
predict.rbm <- function (rbm, newdata)
{
	if (!"rbm" %in% class(rbm))
	{
		message("input object is not an RBM");
		return(NULL);
	}

	if ("integer" %in% class(newdata[1,1]))
	{
		message("Input matrix is Integer: Coercing to Numeric.");
		newdata <- t(apply(newdata, 1, as.numeric));
	}

	#if (!is.loaded("rrbm")) library.dynam("rrbm", package=c("rrbm"), lib.loc=.libPaths());

	.Call("_C_RBM_predict", as.matrix(newdata), as.integer(rbm$n_visible),
		as.integer(rbm$n_hidden), as.matrix(rbm$W), as.numeric(rbm$hbias),
		as.numeric(rbm$vbias),
		PACKAGE = "rrbm");
}

#' Training a Conditional Restricted Boltzmann Machine Function
#'
#' This function trains a CRBM. Returns a CRBM in list form
#' @param dataset A matrix with data, one example per row.
#' @param seqlen A vector with the lenght of sequences in dataset.
#' @param batch_size Number of examples per training mini-batch. Default = 1.
#' @param n_hidden Number of hidden units in the CRBM. Default = 3.
#' @param delay Size of delay window for a sequence. Default = 6.
#' @param training_epochs Number of training epochs. Default = 1000.
#' @param learning_rate The learning rate for training. Default = 0.01.
#' @param momentum The momentum for training. Default = 0.8.
#' @param rand_seed Random seed. Default = 1234.
#' @keywords CRBM, RBM
#' @export
#' @examples
#' train_X <- t(array(c(1, 1, 1, 0, 0, 0, # Sequence 1
#'                      1, 0, 1, 0, 0, 0,
#'                      1, 1, 1, 0, 0, 0,
#'                      0, 0, 0, 1, 1, 0,
#'                      0, 0, 1, 0, 1, 0,
#'                      1, 0, 1, 0, 0, 0,
#'                      1, 0, 1, 0, 1, 0, # Sequence 2
#'                      0, 0, 0, 1, 0, 0,
#'                      0, 0, 0, 0, 1, 0,
#'                      0, 0, 0, 1, 1, 0,
#'                      0, 1, 1, 0, 1, 0,
#'                      1, 0, 1, 1, 1, 0), c(12, 6)));
#' crbm1 <- train.crbm(train_X, seqlen = c(6, 6), delay = 2);
train.crbm <- function (dataset, seqlen, batch_size = 1, n_hidden = 3, delay = 6,
			training_epochs = 1000, learning_rate = 0.01,
			momentum = 0.8, rand_seed = 1234)
{
	if ("integer" %in% class(dataset[1,1]))
	{
		message("Input matrix is Integer: Coercing to Numeric.");
		dataset <- t(apply(dataset, 1, as.numeric));
	}

	#if (!is.loaded("rrbm")) library.dynam("rrbm", package=c("rrbm"), lib.loc=.libPaths());

	retval <- .Call("_C_CRBM_train", as.matrix(dataset), as.integer(seqlen),
		as.integer(length(seqlen)), as.integer(batch_size),
		as.integer(n_hidden), as.integer(training_epochs),
		as.double(learning_rate), as.double(momentum), as.integer(delay),
		as.integer(rand_seed),
		PACKAGE = "rrbm");
	class(retval) <- c("crbm", class(retval));

	retval;
}

#' Predicting using a Conditional Restricted Boltzmann Machine Function
#'
#' Function to Predict data from a CRBM. Returns two matrices: activation
#' and reconstruction.
#' @param crbm A trained CRBM using train.crbm() function.
#' @param newdata A matrix with data, one example per row. Must contain
#' more rows than crbm delay.
#' @keywords CRBM, RBM
#' @export
#' @examples
#' test_X <- t(array(c(1, 1, 0, 0, 0, 0,
#'                     0, 1, 1, 1, 0, 0,
#'                     0, 0, 0, 1, 1, 0), c(6,3)));
#' res <- predict.crbm(crbm1, test_X);
#' ## Also works as
#' res <- predict(crbm1, test_X);
predict.crbm <- function (crbm, newdata)
{
	if (!"crbm" %in% class(crbm))
	{
		message("ERROR: input object is not a CRBM");
		return(NULL);
	}

	if (crbm$delay + 1 > nrow(newdata))
	{
		message("ERROR: Delay is longer than sequence");
		return(NULL);
	}

	if ("integer" %in% class(newdata[1,1]))
	{
		message("ERROR: Input matrix is Integer, Coercing to Numeric.");
		newdata <- t(apply(newdata, 1, as.numeric));
	}

	#if (!is.loaded("rrbm")) library.dynam("rrbm", package=c("rrbm"), lib.loc=.libPaths());

	.Call("_C_CRBM_predict", as.matrix(newdata), as.integer(crbm$n_visible),
		as.integer(crbm$n_hidden), as.matrix(crbm$W), as.matrix(crbm$B),
		as.matrix(crbm$A), as.numeric(crbm$hbias), as.numeric(crbm$vbias),
		as.integer(crbm$delay),
		PACKAGE = "rrbm");
}

#' Predicting using a Conditional Restricted Boltzmann Machine Function
#'
#' Function to, given initialization of visibles and matching history, generate
#' n_samples in future.
#' @param crbm A trained CRBM using train.crbm() function.
#' @param sequence Sequence for first input and its history. A matrix with data,
#' one example per row. Must contain more rows than crbm delay.
#' @param n_samples Number of samples to generate forward. Default = 1.
#' @param n_gibbs Number of alternating Gibbs steps per iteration. Default = 30-
#' @keywords CRBM, RBM, FORECAST
#' @export
#' @examples
#' train_X <- t(array(c(1, 1, 1, 0, 0, 0, # Sequence 1
#'                      1, 0, 1, 0, 0, 0,
#'                      1, 1, 1, 0, 0, 0,
#'                      0, 0, 0, 1, 1, 0,
#'                      0, 0, 1, 0, 1, 0,
#'                      1, 0, 1, 0, 0, 0,
#'                      1, 0, 1, 0, 1, 0, # Sequence 2
#'                      0, 0, 0, 1, 0, 0,
#'                      0, 0, 0, 0, 1, 0,
#'                      0, 0, 0, 1, 1, 0,
#'                      0, 1, 1, 0, 1, 0,
#'                      1, 0, 1, 1, 1, 0), c(12, 6)));
#' crbm1 <- train.crbm(train_X, seqlen = c(6, 6), delay = 2);
#'
#' ## Once trained, predict sequence
#' data_X <- t(array(c(1, 1, 0, 0, 0, 0,
#'                     0, 1, 1, 1, 0, 1,
#'                     0, 1, 0, 1, 0, 0,
#'                     0, 1, 0, 1, 1, 1,
#'                     1, 1, 0, 1, 1, 0,
#'                     1, 1, 0, 0, 0, 0,
#'                     1, 1, 1, 1, 0, 0,
#'                     0, 0, 0, 1, 1, 0), c(6,8)));
#' res3 <- forecast.crbm(crbm1, data_X[1:2, ], 5, 30);
forecast.crbm <- function(crbm, sequence, n_samples = 1, n_gibbs = 30)
{
	if (!"crbm" %in% class(crbm))
	{
		message("ERROR: input object is not a CRBM");
		return(NULL);
	}

	if (crbm$delay + 1 > nrow(sequence))
	{
		message("ERROR: Delay is longer than sequence");
		return(NULL);
	}

	if ("integer" %in% class(sequence[1,1]))
	{
		message("ERROR: Input matrix is Integer, Coercing to Numeric.");
		sequence <- t(apply(sequence, 1, as.numeric));
	}

	#if (!is.loaded("rrbm")) library.dynam("rrbm", package=c("rrbm"), lib.loc=.libPaths());

	.Call("_C_CRBM_generate_samples", as.matrix(sequence), as.integer(crbm$n_visible),
		as.integer(crbm$n_hidden), as.matrix(crbm$W), as.matrix(crbm$B),
		as.matrix(crbm$A), as.numeric(crbm$hbias), as.numeric(crbm$vbias),
		as.integer(crbm$delay), as.integer(n_samples), as.integer(n_gibbs),
		PACKAGE = "rrbm");
}

