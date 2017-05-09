###############################################################################
# Wrapper functions for RBM and CRBM library                                  #
###############################################################################

#' Training a Restricted Boltzmann Machine Function
#'
#' This function trains a RBM. Returns a RBM in list form
#' @param dataset A matrix with data, one example per row.
#' @param batch_size Number of examples per training mini-batch. Default = 1.
#' @param n_hidden Number of hidden units in the RBM. Default = 3.
#' @param training_epochs Number of training epochs. Default = 1000.
#' @param learning_rate The learning rate for training. Default = 0.01.
#' @param momentum The momentum for training. Default = 0.8.
#' @param rand_seed Random seed. Default = 1234.
#' @keywords RBM
#' @export
#' @examples
#' train_X <- t(array(c(1, 1, 1, 0, 0, 0,
#'                      1, 0, 1, 0, 0, 0,
#'                      1, 1, 1, 0, 0, 0,
#'                      0, 0, 1, 1, 1, 0,
#'                      0, 0, 1, 0, 1, 0,
#'                      0, 0, 1, 1, 1, 0), c(6, 6)));
#' rbm1 <- train.rbm(train_X);
#' 
#' # The MNIST example
#' data(mnist)
#'
#' training.num <- data.matrix(mnist$train$x)/255;
#' rbm_mnist <- train.rbm(n_hidden = 30, dataset = training.num,
#'                        learning_rate = 1e-3, training_epochs = 10,
#'                        batch_size = 10, momentum = 0.5);
train.rbm <- function (dataset, batch_size = 1, n_hidden = 3,
			training_epochs = 1000, learning_rate = 0.01,
			momentum = 0.8, rand_seed = 1234)
{
	if ("integer" %in% class(dataset[1,1]))
	{
		message("Input matrix is Integer: Coercing to Numeric.");
		dataset <- t(apply(dataset, 1, as.numeric));
	}


	retval <- .Call("_C_RBM_train", as.matrix(dataset),
		as.integer(batch_size),	as.integer(n_hidden),
		as.integer(training_epochs), as.double(learning_rate),
		as.double(momentum), as.integer(rand_seed),
		PACKAGE = "rrbm");

	class(retval) <- c("rbm", class(retval));

	retval;
}

#' Predicting using a Restricted Boltzmann Machine Function
#'
#' Function to Predict data from a RBM. Returns two matrices: activation
#' and reconstruction.
#' @param rbm A trained RBM using train.rbm() function.
#' @param newdata A matrix with data, one example per row.
#' @keywords RBM
#' @export
#' @examples
#' test_X <- t(array(c(1, 1, 0, 0, 0, 0,
#'                     0, 0, 0, 1, 1, 0), c(6,2)));
#' res <- predict(rbm1, test_X);
#' 
#' # The MNIST example
#' data(mnist)
#'
#' training.num <- data.matrix(mnist$train$x)/255;
#' rbm_mnist <- train.rbm(n_hidden = 30, dataset = training.num,
#'                        learning_rate = 1e-3, training_epochs = 10,
#'                        batch_size = 10, momentum = 0.5);
#'
#' reconstruction <- predict(rbm_mnist, training.num);
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
#'
#' ## Motion (fragment) Example
#' data(motionfrag)
#'
#' crbm_mfrag <- train.crbm(motionfrag$batchdata, motionfrag$seqlen,
#'                          batch_size = 100, n_hidden = 100, delay = 6,
#'                          training_epochs = 200, learning_rate = 1e-3,
#'                          momentum = 0.5, rand_seed = 1234);
train.crbm <- function (dataset, seqlen, batch_size = 1, n_hidden = 3, delay = 6,
			training_epochs = 1000, learning_rate = 0.01,
			momentum = 0.8, rand_seed = 1234)
{
	if ("integer" %in% class(dataset[1,1]))
	{
		message("Input matrix is Integer: Coercing to Numeric.");
		dataset <- t(apply(dataset, 1, as.numeric));
	}

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
#' res <- predict(crbm1, test_X);
#'
#' ## Motion (fragment) Example
#' data(motionfrag)
#'
#' crbm_mfrag <- train.crbm(motionfrag$batchdata, motionfrag$seqlen,
#'                          batch_size = 100, n_hidden = 100, delay = 6,
#'                          training_epochs = 200, learning_rate = 1e-3,
#'                          momentum = 0.5, rand_seed = 1234);
#'
#' preds <- predict(crbm_mfrag, motionfrag$batchdata);
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
#'
#' ## Motion (fragment) Example
#' data(motionfrag)
#'
#' crbm_mfrag <- train.crbm(motionfrag$batchdata, motionfrag$seqlen,
#'                          batch_size = 100, n_hidden = 100, delay = 6,
#'                          training_epochs = 200, learning_rate = 1e-3,
#'                          momentum = 0.5, rand_seed = 1234);
#'
#' offset <- crbm_mfrag$delay + 1;
#' fc <- forecast.crbm(crbm_mfrag, motionfrag$batchdata[1:offset,], 50, 30);
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

	.Call("_C_CRBM_generate_samples", as.matrix(sequence), as.integer(crbm$n_visible),
		as.integer(crbm$n_hidden), as.matrix(crbm$W), as.matrix(crbm$B),
		as.matrix(crbm$A), as.numeric(crbm$hbias), as.numeric(crbm$vbias),
		as.integer(crbm$delay), as.integer(n_samples), as.integer(n_gibbs),
		PACKAGE = "rrbm");
}

