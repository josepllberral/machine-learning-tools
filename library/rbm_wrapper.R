###############################################################################
# Wrapper functions for RBM library                                           #
###############################################################################

dyn.load("rbm.so");

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

## Test function
testing.rbm <- function()
{
	train_X <- t(array(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
		0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0), c(6, 6)));
	rbm1 <- train.rbm(train_X);

	test_X <- t(array(c(1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0), c(6,2)));
	res <- predict(rbm1, test_X);
}



