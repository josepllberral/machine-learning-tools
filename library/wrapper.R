###############################################################################
# Wrapper functions for RBM library                                           #
###############################################################################

dyn.load("library/rbm.so");

## Function to Train a RBM. Returns a RBM in list form
train.rbm <- function (dataset, batch_size = 1, n_hidden = 3,
			training_epochs = 1000, learning_rate = 0.1,
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
		as.double(momentum), as.integer(rand_seed));
	class(retval) <- c("rbm", class(retval));

	retval;
}

## Function to Predict data from a RBM. Returns two matrices:
##   activation and reconstruction
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
		as.numeric(rbm$vbias));
}

## RBM Test function
testing.rbm <- function()
{
	train_X <- t(array(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
		0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0), c(6, 6)));
	rbm1 <- train.rbm(train_X);

	test_X <- t(array(c(1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0), c(6,2)));
	res <- predict(rbm1, test_X);
	res;
}

###############################################################################
# Wrapper functions for CRBM library                                          #
###############################################################################

dyn.load("library/crbm.so");

## Function to Train a CRBM. Returns a CRBM in list form
train.crbm <- function (dataset, seqlen, batch_size = 1, n_hidden = 3, delay = 6,
			training_epochs = 1000, learning_rate = 0.1,
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
		as.integer(rand_seed));
	class(retval) <- c("crbm", class(retval));

	retval;
}

## Function to Predict data from a RBM. Returns two matrices:
##   activation and reconstruction
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
		as.integer(crbm$delay));
}

## Given initialization(s) of visibles and matching history, generate n_samples in future.
##	sequence  : n_seq by n_visibles array, sequence for first input and its history
##	n_samples : int, number of samples to generate forward
##	n_gibbs   : int, number of alternating Gibbs steps per iteration
forecast.crbm <- function(crbm, sequence, n_samples, n_gibbs = 30)
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
		as.integer(crbm$delay), as.integer(n_samples), as.integer(n_gibbs));
}


testing.crbm <- function()
{
	dataset <- load_data('./datasets/motion.rds');

	crbm <- train.crbm (dataset$batchdata, dataset$seqlen, batch_size = 100, n_hidden = 100, delay = 6, training_epochs = 200, learning_rate = 1e-3, momentum = 0.5, rand_seed = 1234);

	res1 <- predict(crbm, dataset$batchdata);
	res2 <- forecast.crbm(crbm, dataset$batchdata[1:(crbm$delay+1),], 50, 30);

	list(predicted = res1, forecast = res2);
}

