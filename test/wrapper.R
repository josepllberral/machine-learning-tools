###############################################################################
# Wrapper functions for CNN library                                           #
###############################################################################

## Function to produce a confusion matrix
conf.matrix <- function(a, b)
{
	if (all.equal(dim(a), dim(b)) != TRUE)
	{
		message("ERROR: Matrices have different dimensions");
		return (NULL);
	}

	confusion <- matrix(0, ncol(a), ncol(a));

	for (i in 1:nrow(a))
	{
		rowa <- a[i,];
		rowb <- b[i,];

		sel_a <- which(rowa == max(rowa));
		sel_b <- which(rowb == max(rowb));

		confusion[sel_a, sel_b] <- confusion[sel_a, sel_b] + 1;
	}

	confusion;
}

## Function to binarize vectors
binarization <- function(vec)
{
	result <- array(0, c(length(vec),length(unique(vec))));
	for (i in 1:length(vec)) result[i,vec[i]] <- 1;
	result;
}

## Function to check the basic shape of data between Layers
check_layers <- function (layers, dataset, target, batch_size)
{
	# Input Dimensions
	nrow <- dim(dataset)[1];
	ncol <- dim(dataset)[2];
	if (length(dim(dataset)) == 4)
	{
		img_h <- dim(dataset)[3];
		img_w <- dim(dataset)[4];

		input_dims <- c(batch_size, dim(dataset)[2:4]);
	} else if (length(dim(dataset)) == 2)
	{
		input_dims <- c(batch_size, dim(dataset)[2]);
	}

	nrow_y <- dim(target)[1];
	ncol_y <- dim(target)[2];

	# Check inputs vs outputs
	if (nrow != nrow_y)
	{
		message("Error in Inputs");
		message("Inputs and Output rows do not match");
		return (-1);
	}

	nlayers <- length(layers);

	# Check Pipeline
	for (i in 1:nlayers)
	{
		laux <- layers[[i]];

		# Check for valid values
		pass <- all(!is.na(as.numeric(laux[-1])));
		pass <- pass && all(as.numeric(laux[-1]) > 0);
		if (!pass)
		{
			message(paste("Error in layer ", i, sep = ""));
			message("Incorrect input value (negative, character or zero...?)");
			return (FALSE);
		}

		# Check for Layers
		if (laux[1] == "CONV")
		{
			# Check for Batch_size and Channels
			if (all.equal(input_dims[1:2], as.numeric(laux[c(7,2)])) != TRUE)
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current CONV input (batch_size, channels) do not match previous LAYER output (batch_size, channels)");
				return (FALSE);
			}
			input_dims[2] <- as.numeric(laux[3]);
		} else if (laux[1] == "POOL")
		{
			# Check for Batch_size and Channels
			if (all.equal(input_dims[1:2], as.numeric(laux[c(4,2)])) != TRUE)
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current POOL input (batch_size, channels) do not match previous LAYER output (batch_size, channels)");
				return (FALSE);
			}
			out_h <- (input_dims[3] - as.numeric(laux[5]) + 2 * as.numeric(laux[5]) %/% 2) %/% as.numeric(laux[6]) + 1;
			out_w <- (input_dims[4] - as.numeric(laux[5]) + 2 * as.numeric(laux[5]) %/% 2) %/% as.numeric(laux[6]) + 1;
			input_dims[3:4] <- c(out_h, out_w);
		} else if (laux[1] == "RELU")
		{
			# Check for Batch_size and Channels
			if (all.equal(input_dims[1:2], as.numeric(laux[3:2])) != TRUE)
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current RELU input (batch_size, channels) do not match previous LAYER output (batch_size, channels)");
				return (FALSE);
			}
		} else if (laux[1] == "FLAT")
		{
			# Check for Batch_size and Channels
			if (all.equal(input_dims[1:2], as.numeric(laux[3:2])) != TRUE)
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current FLAT input (batch_size, channels) do not match previous LAYER output (batch_size, channels)");
				return (FALSE);
			}
			input_dims <- c(input_dims[1], input_dims[2] * input_dims[3] * input_dims[4]);
		} else if (laux[1] == "LINE")
		{
			# Check for Batch_size and Visible units
			if (all.equal(input_dims, as.numeric(laux[c(5,2)])) != TRUE)
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current LINE input (batch_size, visible) do not match previous LAYER output (batch_size, visible)");
				return (FALSE);
			}
			input_dims[2] <- as.numeric(laux[3]);
		} else if (laux[1] == "RELV")
		{
			# Check for Batch_size
			if (input_dims[1] != as.numeric(laux[2]))
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current RELV input (batch_size) do not match previous LAYER output (batch_size)");
				return (FALSE);
			}
		} else if (laux[1] %in% c("SOFT", "SIGM", "TANH"))
		{
			# Check for Batch_size and Visible units
			if (all.equal(input_dims, as.numeric(laux[3:2])) != TRUE)
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current SOFT/SIGM/TANH input (batch_size, visible) do not match previous LAYER output (batch_size, visible)");
				return (FALSE);
			}
		} else if (laux[1] == "DIRE")
		{
			# Check for Batch_size
			if (input_dims[1] != as.numeric(laux[2]))
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current DIRE input (batch_size) do not match previous LAYER output (batch_size)");
				return (FALSE);
			}
		}
	}

	if (!layers[[nlayers]][1] %in% c("SOFT","SIGM","LINE","DIRE","TANH"))
	{
		message("Error in Output Layer");
		message("Output layer must be a SOFT, SIGM, TANH, LINE or DIRE");
		return (FALSE);
	}

	# Check Output
	if (all.equal(input_dims, c(batch_size, ncol_y)) != TRUE)
	{
		message("Error in Output Data");
		message("Output data does not match with network output");
		return (FALSE);
	}

	return (TRUE);
}

#' Training a Convolutional Neural Network Function
train.cnn <- function (dataset, targets, layers,  batch_size = 10,
			training_epochs = 10, learning_rate = 1e-3,
			momentum = 0.8, rand_seed = 1234)
{
	if ("integer" %in% class(dataset[1,1,1,1]))
	{
		message("Input matrix is Integer: Coercing to Numeric.");
		dataset <- t(apply(dataset, c(1,2,3), as.numeric));
	}

	if (!check_layers(layers, dataset, targets, batch_size))
	{
		message("Network does not match with data dimensions");
		return(NULL);
	}

	#if (!is.loaded("bscnn")) library.dynam("bscnn", package=c("bscnn"), lib.loc=.libPaths());

	retval <- .Call("_C_CNN_train", as.array(dataset), as.matrix(targets),
		as.list(layers), as.integer(length(layers)), as.integer(batch_size),
		as.integer(training_epochs), as.double(learning_rate), as.double(momentum),
		as.integer(rand_seed));#, PACKAGE = "rcnn");

	class(retval) <- c("cnn", class(retval));

	retval;
}

#' Predicting using a Convolutional Network Function
predict.cnn <- function (cnn, newdata)
{
	if (!"cnn" %in% class(cnn))
	{
		message("input object is not a CNN");
		return(NULL);
	}

	if ("integer" %in% class(newdata[1,1,1,1]))
	{
		message("Input matrix is Integer: Coercing to Numeric.");
		newdata <- t(apply(newdata, c(1,2,3), as.numeric));
	}

	#if (!is.loaded("bscnn")) library.dynam("bscnn", package=c("bscnn"), lib.loc=.libPaths());

	.Call("_C_CNN_predict", as.array(newdata), as.list(cnn$layers),
		as.integer(length(cnn$layers))); #PACKAGE = "rcnn");
}

#' Training a MultiLayer Perceptron Neural Network Function
train.mlp <- function (dataset, targets, layers,  batch_size = 10,
			training_epochs = 10, learning_rate = 1e-3,
			momentum = 0.8, rand_seed = 1234)
{
	if ("integer" %in% class(dataset[1,1]))
	{
		message("Input matrix is Integer: Coercing to Numeric.");
		dataset <- t(apply(dataset, 1, as.numeric));
	}

	if (!check_layers(layers, dataset, targets, batch_size))
	{
		message("Network does not match with data dimensions");
		return(NULL);
	}

	# Check that all layers are Matricial
	for (i in 1:length(layers))
		if (!(layers[[i]][1] %in% c("LINE","RELV","SOFT","SIGM","TANH","DIRE")))
		{
			message(paste("Layers contain non-matricial layer", layers[[i]][1], "at", i, sep = " "));
			return(NULL);
		}

	#if (!is.loaded("bscnn")) library.dynam("bscnn", package=c("bscnn"), lib.loc=.libPaths());

	retval <- .Call("_C_MLP_train", as.matrix(dataset), as.matrix(targets),
		as.list(layers), as.integer(length(layers)), as.integer(batch_size),
		as.integer(training_epochs), as.double(learning_rate), as.double(momentum),
		as.integer(rand_seed));#, PACKAGE = "rcnn");

	class(retval) <- c("mlp", class(retval));

	retval;
}

#' Predicting using a Convolutional Network Function
predict.mlp <- function (mlp, newdata)
{
	if (!"mlp" %in% class(mlp))
	{
		message("input object is not a MLP");
		return(NULL);
	}

	if ("integer" %in% class(newdata[1,1]))
	{
		message("Input matrix is Integer: Coercing to Numeric.");
		newdata <- t(apply(newdata, 1, as.numeric));
	}

	#if (!is.loaded("bscnn")) library.dynam("bscnn", package=c("bscnn"), lib.loc=.libPaths());

	.Call("_C_MLP_predict", as.matrix(newdata), as.list(mlp$layers),
		as.integer(length(mlp$layers))); #PACKAGE = "rcnn");
}

