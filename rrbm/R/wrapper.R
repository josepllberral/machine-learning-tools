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
#' @param init_rbm Initial values for RBM from a trained one. Default = NULL.
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
#' rbm1_update <- train.rbm(train_X, init_rbm = rbm1);
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
			momentum = 0.8, rand_seed = 1234, init_rbm = NULL)
{
	if (is.null(dataset))
	{
		message("The input dataset is NULL");
		return(NULL);
	}
	
	if (is.null(batch_size) || is.null(n_hidden) || is.null(training_epochs)
	|| is.null(learning_rate) || is.null(momentum) || is.null(rand_seed))
	{
		message("Some mandatory parameters are NULL");
		return(NULL);
	}

	if ("integer" %in% class(dataset[1,1]))
	{
		message("Input matrix is Integer: Coercing to Numeric.");
		dataset <- t(apply(dataset, 1, as.numeric));
	}

	if (!is.null(init_rbm))
	{
		if (!"rbm" %in% class(init_rbm))
		{
			message("input object is not an RBM");
			return(NULL);
		}

		if (dim(init_rbm$W) != c(ncol(dataset), n_hidden) ||
			length(init_rbm$hbias) != n_hidden ||
			length(init_rbm$vbias) != ncol(dataset))
		{
			message("Dimensions of initial RBM do not match");
			return(NULL);
		}
		
		retval <- .Call("_C_RBM_update", as.matrix(dataset),
			as.integer(batch_size),	as.integer(n_hidden),
			as.integer(training_epochs), as.double(learning_rate),
			as.double(momentum), as.integer(rand_seed),
			as.matrix(init_rbm$W), as.numeric(init_rbm$hbias),
			as.numeric(init_rbm$vbias),
			PACKAGE = "rrbm");
	} else {	
		retval <- .Call("_C_RBM_train", as.matrix(dataset),
			as.integer(batch_size),	as.integer(n_hidden),
			as.integer(training_epochs), as.double(learning_rate),
			as.double(momentum), as.integer(rand_seed),
			PACKAGE = "rrbm");
	}
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
#' passed_data <- predict(rbm_mnist, training.num);
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

#' Featurizing using a Restricted Boltzmann Machine Function
#'
#' Function to Activate feaures from a RBM (Forward pass). Returns an
#' activation matrix.
#' @param rbm A trained RBM using train.rbm() function.
#' @param act.input A matrix with activation data, one example per row.
#' @keywords RBM
#' @export
#' @examples
#' test_X <- t(array(c(1, 1, 0, 0, 0, 0,
#'                     0, 0, 0, 1, 1, 0), c(6,2)));
#' res <- predict(rbm1, test_X);
#' reconstruction <- backward.rbm(rbm1, res$activation);
#' 
#' activation <- forward.rbm(rbm1, test_X);
#' reconstruction <- backward.rbm(rbm1, activation);
#'
#' # The MNIST example
#' data(mnist)
#'
#' training.num <- data.matrix(mnist$train$x)/255;
#' rbm_mnist <- train.rbm(n_hidden = 30, dataset = training.num,
#'                        learning_rate = 1e-3, training_epochs = 10,
#'                        batch_size = 10, momentum = 0.5);
#'
#' passed_data <- predict(rbm_mnist, training.num);
#' reconstruction <- backward.rbm(rbm_mnist, passed_data$activation);
#'
#' activation <- forward.rbm(rbm_mnist, training.num);
#' reconstruction <- backward.rbm(rbm_mnist, activation);
forward.rbm <- function (rbm, newdata)
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

	.Call("_C_RBM_forward", as.matrix(newdata), as.integer(rbm$n_visible),
		as.integer(rbm$n_hidden), as.matrix(rbm$W), as.numeric(rbm$hbias),
		as.numeric(rbm$vbias),
		PACKAGE = "rrbm");
}

#' Reconstructing using a Restricted Boltzmann Machine Function
#'
#' Function to Reconstruct data from a RBM (Backward pass). Returns a
#' reconstruction matrix.
#' @param rbm A trained RBM using train.rbm() function.
#' @param act.input A matrix with activation data, one example per row.
#' @keywords RBM
#' @export
#' @examples
#' test_X <- t(array(c(1, 1, 0, 0, 0, 0,
#'                     0, 0, 0, 1, 1, 0), c(6,2)));
#' res <- predict(rbm1, test_X);
#' reconstruction <- backward.rbm(rbm1, res$activation);
#' 
#' activation <- forward.rbm(rbm1, test_X);
#' reconstruction <- backward.rbm(rbm1, activation);
#'
#' # The MNIST example
#' data(mnist)
#'
#' training.num <- data.matrix(mnist$train$x)/255;
#' rbm_mnist <- train.rbm(n_hidden = 30, dataset = training.num,
#'                        learning_rate = 1e-3, training_epochs = 10,
#'                        batch_size = 10, momentum = 0.5);
#'
#' passed_data <- predict(rbm_mnist, training.num);
#' reconstruction <- backward.rbm(rbm_mnist, passed_data$activation);
#'
#' activation <- forward.rbm(rbm_mnist, training.num);
#' reconstruction <- backward.rbm(rbm_mnist, activation);
backward.rbm <- function (rbm, act.input)
{
	if (!"rbm" %in% class(rbm))
	{
		message("input object is not an RBM");
		return(NULL);
	}

	if ("integer" %in% class(act.input[1,1]))
	{
		message("Input matrix is Integer: Coercing to Numeric.");
		newdata <- t(apply(act.input, 1, as.numeric));
	}

	.Call("_C_RBM_backward", as.matrix(act.input), as.integer(rbm$n_visible),
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
#' @param init_crbm Initial values for CRBM from a trained one. Default = NULL.
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
#'                      1, 0, 1, 1, 1, 0), c(6, 12)));
#' crbm1 <- train.crbm(train_X, seqlen = c(6, 6), delay = 2);
#' crbm1_update <- train.crbm(train_X, seqlen = c(6, 6), delay = 2,
#'                            init_crbm = crbm1);
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
			momentum = 0.8, rand_seed = 1234, init_crbm = NULL)
{
	if (is.null(dataset) || is.null(seqlen))
	{
		message("The input dataset or seqlen are NULL");
		return(NULL);
	}
	
	if (is.null(batch_size) || is.null(n_hidden) || is.null(training_epochs)
	|| is.null(learning_rate) || is.null(momentum) || is.null(rand_seed)
	|| is.null(delay))
	{
		message("Some mandatory parameters are NULL");
		return(NULL);
	}

	if ("integer" %in% class(dataset[1,1]))
	{
		message("Input matrix is Integer: Coercing to Numeric.");
		dataset <- t(apply(dataset, 1, as.numeric));
	}

	if (sum(as.integer(seqlen)) != nrow(dataset))
	{
		message("Sequence Lenght vector does not sum the total Dataset Rows");
		return(NULL);
	}

	if (!is.null(init_crbm))
	{
		if (!"crbm" %in% class(init_crbm))
		{
			message("input object is not a CRBM");
			return(NULL);
		}

		if (dim(init_crbm$W) != c(ncol(dataset), n_hidden) ||
			dim(init_crbm$B) != c(ncol(dataset) * delay, n_hidden) ||
			dim(init_crbm$A) != c(ncol(dataset) * delay, ncol(dataset)) ||
			length(init_crbm$hbias) != n_hidden ||
			length(init_crbm$vbias) != ncol(dataset) ||
			init_crbm$delay != delay
		)
		{
			message("Dimensions of initial CRBM do not match");
			return(NULL);
		}
		
		retval <- .Call("_C_CRBM_update", as.matrix(dataset), as.integer(seqlen),
			as.integer(length(seqlen)), as.integer(batch_size),
			as.integer(n_hidden), as.integer(training_epochs),
			as.double(learning_rate), as.double(momentum), as.integer(delay),
			as.integer(rand_seed), as.matrix(init_crbm$W), as.matrix(init_crbm$B),
			as.matrix(init_crbm$A), as.numeric(init_crbm$hbias), as.numeric(init_crbm$vbias),
			PACKAGE = "rrbm");
	} else {
		retval <- .Call("_C_CRBM_train", as.matrix(dataset), as.integer(seqlen),
			as.integer(length(seqlen)), as.integer(batch_size),
			as.integer(n_hidden), as.integer(training_epochs),
			as.double(learning_rate), as.double(momentum), as.integer(delay),
			as.integer(rand_seed),
			PACKAGE = "rrbm");
	}
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

#' Feature generation using a Conditional Restricted Boltzmann Machine Function
#'
#' Function to Generate features from a CRBM (Forward pass). Returns a
#' prediction matrix.
#' @param crbm A trained CRBM using train.crbm() function.
#' @param newdata A matrix with data, one example per row. Must contain
#' more rows than crbm delay.
#' @keywords CRBM, RBM
#' @export
#' @examples
#' test_X <- t(array(c(1, 1, 0, 0, 0, 0,
#'                     0, 1, 1, 1, 0, 0,
#'                     0, 0, 0, 1, 1, 0), c(6,3)));
#' activations <- forward.crbm((crbm1, test_X);
#'
#' ## Motion (fragment) Example
#' data(motionfrag)
#'
#' crbm_mfrag <- train.crbm(motionfrag$batchdata, motionfrag$seqlen,
#'                          batch_size = 100, n_hidden = 100, delay = 6,
#'                          training_epochs = 200, learning_rate = 1e-3,
#'                          momentum = 0.5, rand_seed = 1234);
#'
#' activations <- forward.crbm(crbm_mfrag, motionfrag$batchdata);
forward.crbm <- function (crbm, newdata)
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

	.Call("_C_CRBM_forward", as.matrix(newdata), as.integer(crbm$n_visible),
		as.integer(crbm$n_hidden), as.matrix(crbm$W), as.matrix(crbm$B),
		as.matrix(crbm$A), as.numeric(crbm$hbias), as.numeric(crbm$vbias),
		as.integer(crbm$delay),
		PACKAGE = "rrbm");
}

#' Reconstruction using a Conditional Restricted Boltzmann Machine Function
#'
#' Function to reconstruct inputs from a CRBM (Forward pass). Returns a
#' prediction matrix.
#' @param crbm A trained CRBM using train.crbm() function.
#' @param newdata A matrix with data, one example per row. Must contain
#' more rows than crbm delay.
#' @param history A matrix with the visible history data, one example per row.
#' Must contain as many rows as newdata plus the delay
#' @keywords CRBM, RBM
#' @export
#' @examples
#' test_X <- t(array(c(1, 1, 0, 0, 0, 0,
#'                     0, 1, 1, 1, 0, 0,
#'                     0, 0, 0, 1, 1, 0), c(6,3)));
#' activations <- forward.crbm((crbm1, test_X);
#' reconstruct <- backward.crbm((crbm1, activations, test_X); # FIXME
#'
#' ## Motion (fragment) Example
#' data(motionfrag)
#'
#' crbm_mfrag <- train.crbm(motionfrag$batchdata, motionfrag$seqlen,
#'                          batch_size = 100, n_hidden = 100, delay = 6,
#'                          training_epochs = 200, learning_rate = 1e-3,
#'                          momentum = 0.5, rand_seed = 1234);
#'
#' activations <- forward.crbm(crbm_mfrag, motionfrag$batchdata);
#' 
#' valid_activations <- activations[crbm_mfrag$delay:nrow(activations)];
#' corresp_history <- motionfrag$batchdata[1:(nrow(motionfrag$batchdata)-1),];
#' reconstruct <- backward.crbm(crbm_mfrag, valid_activations, corresp_history);
backward.crbm <- function (crbm, newdata, history)
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

	if (nrow(history) != nrow(newdata) + crbm$delay - 1)
	{
		message("ERROR: History dataset rows not match Activation dataset rows + delay size");
		return(NULL);
	}

	if ("integer" %in% class(newdata[1,1]))
	{
		message("ERROR: Input matrix is Integer, Coercing to Numeric.");
		newdata <- t(apply(newdata, 1, as.numeric));
	}

	.Call("_C_CRBM_backward", as.matrix(newdata), as.matrix(history),
		as.integer(crbm$n_visible), as.integer(crbm$n_hidden),
		as.matrix(crbm$W), as.matrix(crbm$B), as.matrix(crbm$A),
		as.numeric(crbm$hbias), as.numeric(crbm$vbias),
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
