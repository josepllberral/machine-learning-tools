################################################################################
# Wrapper functions for CNN/MLP library                                        #
################################################################################

#' Function to produce a confusion matrix
#'
#' This function creates a confusion matrix from two vectors/matrices with
#' output values. If the matrices are not "one-hot encoded", it takes the
#' maximum value per row as example output. The first matrix value is
#' represented as "rows" while the second matrix is represented as "columns".
#' @param a First matrix to compare
#' @param b Second matrix to be compared
#' @keywords Confusion-Matrix
#' @export
#' @examples
#' vector1 <- as.integer((rnorm(1000, 0, 1)  %% 1) * 10);
#' targets <- binarize(vector1);
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

#' Function to binarize vectors
#'
#' This function creates a one-hot encoded matrix given a vector.
#' @param vec A vector of integers
#' @keywords One-Hot-Encode Binarization
#' @export
#' @examples
#' vector1 <- as.integer((rnorm(1000, 0, 1)  %% 1) * 10);
#' targets <- binarize(vector1);
binarization <- function(vec)
{
	if (!is.integer(vec))
	{
		return(NULL);
	}
	result <- array(0, c(length(vec),length(unique(vec))));
	for (i in 1:length(vec)) result[i,vec[i]] <- 1;
	result;
}

#' Function to check the basic shape of data between Layers
#'
#' This function checks if a list of layer descriptors is correctly dimensioned.
#' @param layers A list of descriptors
#' @param dataset A matrix with data, one example per row.
#' @param target A matrix with data targets, one example per row.
#' @param batch_size The batch_size to be used in the CNN or MLP
#' @return TRUE if the list input and output dimensions match. Otherwise returns
#' FALSE.
#' @keywords MLP
#' @export
#' @examples
#' dataset <- array(1:784000, c(1000, 1, 28, 28));
#' targets <- binarize(as.integer((rnorm(1000, 0, 1)  %% 1) * 10));
#' layers <- list(
#'              c("CONV", 1, 4, filter_size, 0.1, border_mode, batch_size),
#'              c("POOL", 4, 0.1, batch_size, win_size, stride),
#'              c("RELU", 4, batch_size),
#'              c("CONV", 4, 16, filter_size, 0.1, border_mode, batch_size),
#'              c("POOL", 16, 0.1, batch_size, win_size, stride),
#'              c("RELU", 16, batch_size),
#'              c("FLAT", 16, batch_size),
#'              c("LINE", 784, 64, 0.1, batch_size),
#'              c("RELV", batch_size),
#'              c("LINE", 64, 10, 0.1, batch_size),
#'              c("SOFT", 10, batch_size)
#' );
#' batch_size <- 10;
#' is_valid <- check_layers(layers, dataset, targets, batch_size);
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
		message(paste("Error in Inputs. Dataset:", nrow, "Target:", nrow_y, sep = " "));
		message("Inputs and Output rows do not match");
		return (FALSE);
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
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
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
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
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
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
				return (FALSE);
			}
		} else if (laux[1] == "FLAT")
		{
			# Check for Batch_size and Channels
			if (all.equal(input_dims[1:2], as.numeric(laux[3:2])) != TRUE)
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current FLAT input (batch_size, channels) do not match previous LAYER output (batch_size, channels)");
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
				return (FALSE);
			}
			input_dims <- c(input_dims[1], input_dims[2] * input_dims[3] * input_dims[4]); #FIXME
		} else if (laux[1] == "LINE")
		{
			# Check for Batch_size and Visible units
			if (all.equal(input_dims, as.numeric(laux[c(5,2)])) != TRUE)
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current LINE input (batch_size, visible) do not match previous LAYER output (batch_size, visible)");
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
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
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
				return (FALSE);
			}
		} else if (laux[1] %in% c("SOFT", "SIGM", "TANH"))
		{
			# Check for Batch_size and Visible units
			if (all.equal(input_dims, as.numeric(laux[3:2])) != TRUE)
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current SOFT/SIGM/TANH input (batch_size, visible) do not match previous LAYER output (batch_size, visible)");
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
				return (FALSE);
			}
		} else if (laux[1] == "DIRE")
		{
			# Check for Batch_size
			if (input_dims[1] != as.numeric(laux[2]))
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current DIRE input (batch_size) do not match previous LAYER output (batch_size)");
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
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
#'
#' This function trains a CNN. Admits as parameters the training dataset, the
#' matrix of outputs, and a descriptor of the network including all the layers
#' and their properties. Returns a CNN in list form, including all the trained
#' layers.
#' Possible layers are:
#' \itemize{
#'   \item CONV: Convolutional Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of channels of the input image
#'     \item Number of filters to be returned
#'     \item Size of convolutional filters
#'     \item Scale for initialization weights
#'     \item Border Mode (1 = valid, 2 = same, 3 = full)
#'     \item Batch_size
#'   }
#'   \item POOL: Pooling Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of channels of the input image.
#'     \item Scale for initialization weights
#'     \item Batch_size
#'     \item Window size
#'     \item Stride
#'   }
#'   \item RELU: Rectified Linear. Requires, in the following order:
#'   \enumerate{
#'     \item Number of channels of the input image.
#'     \item Batch_size
#'   }
#'   \item FLAT: Flattening Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of channels of the input image.
#'     \item Batch_size
#'   }
#'   \item LINE: Linear Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'     \item Number of hidden units
#'     \item Scale for initialization weights
#'     \item Batch_size
#'   }
#'   \item RELV: Rectified Linear (for flattened batches). Requires, in the
#'   following order:
#'   \enumerate{
#'     \item Batch_size
#'   }
#'   \item SOFT: SoftMax Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'     \item Batch_size
#'   }
#'   \item SIGM: Sigmoid Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'     \item Batch_size
#'   }
#'   \item TANH: Hyperbolic Tangent Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'     \item Batch_size
#'   }
#'   \item DIRE: Direct (buffer) Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'     \item Batch_size
#'   }
#' }
#' The attribute "batch_size" must be the same for all layers. The list of
#' layers is checked before the process starts.
#' @param dataset A matrix with data, one example per row.
#' @param targets A matrix with output labels, one set of targets per row.
#' @param batch_size Number of examples per training mini-batch. Default = 1.
#' @param training_epochs Number of training epochs. Default = 1000.
#' @param learning_rate The learning rate for training. Default = 0.01.
#' @param momentum The momentum for training. Default = 0.8. (Not Implemented!)
#' @param rand_seed Random seed. Default = 1234.
#' @keywords CNN
#' @export
#' @examples
#' ## Simple example
#' train_X <- array(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
#'                    0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0),
#'                    c(6, 1, 2, 3));
#' train_Y <- array(c(1, 0,
#'                    1, 0,
#'                    1, 1,
#'                    0, 0,
#'                    0, 1,
#'                    0, 1), c(6, 2));
#'
#' batch_size <- 2;
#' filter_size <- 5;
#' border_mode <- 2; # "SAME" mode
#' win_size <- 3;
#' stride <- 2;
#'
#' layers <- list(
#'    c("CONV", 1, 4, filter_size, 0.1, border_mode, batch_size),
#'    c("POOL", 4, 0.1, batch_size, win_size, stride),
#'    c("RELU", 4, batch_size),
#'    c("FLAT", 4, batch_size),
#'    c("LINE", 6, 2, 0.1, batch_size),
#'    c("SOFT", 2, batch_size)
#' );
#'
#' cnn1 <- train.cnn(train_X, train_Y, layers);
#'
#' ## The MNIST example
#' data(mnist)
#'
#' img_size <- c(28,28);
#'
#' train <- mnist$train;
#' training_x <- array(train$x, c(nrow(train$x), 1, img_size)) / 255;
#' training_y <- binarization(train$y);
#'
#' test <- mnist$test;
#' testing_x <- array(test$x, c(nrow(test$x), 1, img_size)) / 255;
#' testing_y <- binarization(test$y);
#'
#' dataset <- training_x[1:1000,,,, drop=FALSE];
#' targets <- training_y[1:1000,, drop=FALSE];
#'
#' newdata <- testing_x[1:1000,,,, drop=FALSE];
#'
#' batch_size <- 10;
#' training_epochs <- 3;
#' learning_rate <- 1e-3;
#' momentum <- 0.8;
#' rand_seed <- 1234;
#' border_mode <- 2; # "SAME" mode
#' filter_size <- 5;
#' win_size <- 3;
#' stride <- 2;
#'
#' layers <- list(
#'              c("CONV", 1, 4, filter_size, 0.1, border_mode, batch_size),
#'              c("POOL", 4, 0.1, batch_size, win_size, stride),
#'              c("RELU", 4, batch_size),
#'              c("CONV", 4, 16, filter_size, 0.1, border_mode, batch_size),
#'              c("POOL", 16, 0.1, batch_size, win_size, stride),
#'              c("RELU", 16, batch_size),
#'              c("FLAT", 16, batch_size),
#'              c("LINE", 784, 64, 0.1, batch_size),
#'              c("RELV", batch_size),
#'              c("LINE", 64, 10, 0.1, batch_size),
#'              c("SOFT", 10, batch_size)
#' );
#'
#' mnist_cnn <- train.cnn(dataset, targets, layers, batch_size, training_epochs,
#'                        learning_rate, momentum, rand_seed);
train.cnn <- function (dataset, targets, layers,  batch_size = 10,
			training_epochs = 10, learning_rate = 1e-3,
			momentum = 0.8, rand_seed = 1234)
{
	if (is.null(dataset) || is.null(targets) || is.null(layers))
	{
		message("The input dataset, targets or layers are NULL");
		return(NULL);
	}
	
	if (is.null(batch_size) || is.null(training_epochs)
	|| is.null(learning_rate) || is.null(momentum) | is.null(rand_seed))
	{
		message("Some mandatory parameters are NULL");
		return(NULL);
	}

	if ("integer" %in% class(dataset[1,1,1,1]))
	{
		message("Input matrix is Integer: Coercing to Numeric.");
		dataset <- 1.0 * dataset;
	}

# TODO - Change check_layers by descriptor through names

	if (!check_layers(layers, dataset, targets, batch_size))
	{
		message("Network does not match with data dimensions");
		return(NULL);
	}

# TODO - Rearrange layers descriptor through name

	retval <- .Call("_C_CNN_train", as.array(dataset), as.matrix(targets),
		as.list(layers), as.integer(length(layers)), as.integer(batch_size),
		as.integer(training_epochs), as.double(learning_rate), as.double(momentum),
		as.integer(rand_seed), PACKAGE = "rcnn");

	class(retval) <- c("cnn", class(retval));

	retval;
}

#' Predicting using a Convolutional Network Function
#'
#' This function predicts a dataset using a trained CNN. Admits as parameters
#' the testing dataset, and a trained CNN. Returns a matrix of predicted
#' outputs.
#' @param cnn A trained CNN.
#' @param newdata A matrix with data, one example per row.
#' @keywords CNN
#' @export
#' @examples
#' ## Simple example
#' test_X <- array(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
#'                   0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0),
#'                   c(6, 1, 2, 3));
#' results <- predict(cnn1, test_X);
#'
#' ## The MNIST example
#' data(mnist)
#'
#' img_size <- c(28,28);
#'
#' train <- mnist$train;
#' training_x <- array(train$x, c(nrow(train$x), 1, img_size)) / 255;
#' training_y <- binarization(train$y);
#'
#' test <- mnist$test;
#' testing_x <- array(test$x, c(nrow(test$x), 1, img_size)) / 255;
#' testing_y <- binarization(test$y);
#'
#' dataset <- training_x[1:1000,,,, drop=FALSE];
#' targets <- training_y[1:1000,, drop=FALSE];
#'
#' newdata <- testing_x[1:1000,,,, drop=FALSE];
#'
#' batch_size <- 10;
#' training_epochs <- 3;
#' learning_rate <- 1e-3;
#' momentum <- 0.8;
#' rand_seed <- 1234;
#' border_mode <- 2; # "SAME" mode
#' filter_size <- 5;
#' win_size <- 3;
#' stride <- 2;
#'
#' layers <- list(
#'              c("CONV", 1, 4, filter_size, 0.1, border_mode, batch_size),
#'              c("POOL", 4, 0.1, batch_size, win_size, stride),
#'              c("RELU", 4, batch_size),
#'              c("CONV", 4, 16, filter_size, 0.1, border_mode, batch_size),
#'              c("POOL", 16, 0.1, batch_size, win_size, stride),
#'              c("RELU", 16, batch_size),
#'              c("FLAT", 16, batch_size),
#'              c("LINE", 784, 64, 0.1, batch_size),
#'              c("RELV", batch_size),
#'              c("LINE", 64, 10, 0.1, batch_size),
#'              c("SOFT", 10, batch_size)
#' );
#'
#' mnist_cnn <- train.cnn(dataset, targets, layers, batch_size, training_epochs,
#'                        learning_rate, momentum, rand_seed);
#'
#' prediction <- predict(mnist_cnn, newdata);
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
		newdata <- 1.0 * newdata;
	}

	.Call("_C_CNN_predict", as.array(newdata), as.list(cnn$layers),
		as.integer(length(cnn$layers)), PACKAGE = "rcnn");
}

#' Training a MultiLayer Perceptron Neural Network Function
#'
#' This function trains a MLP ANN. Admits as parameters the training dataset,
#' the matrix of outputs, and a descriptor of the network including all the
#' layers and their properties. Returns a MLP in list form, including all the
#' trained layers.
#' Possible layers are:
#' \itemize{
#'   \item LINE: Linear Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'     \item Number of hidden units
#'     \item Scale for initialization weights
#'     \item Batch_size
#'   }
#'   \item RELV: Rectified Linear (for flattened batches). Requires, in the following order:
#'   \enumerate{
#'     \item Batch_size
#'   }
#'   \item SOFT: SoftMax Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'     \item Batch_size
#'   }
#'   \item SIGM: Sigmoid Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'     \item Batch_size
#'   }
#'   \item TANH: Hyperbolic Tangent Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'     \item Batch_size
#'   }
#'   \item DIRE: Direct (buffer) Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'     \item Batch_size
#'   }
#' }
#' The attribute "batch_size" must be the same for all layers. The list of
#' layers is checked before the process starts.
#' @param dataset A matrix with data, one example per row.
#' @param targets A matrix with output labels, one set of targets per row.
#' @param batch_size Number of examples per training mini-batch. Default = 1.
#' @param training_epochs Number of training epochs. Default = 1000.
#' @param learning_rate The learning rate for training. Default = 0.01.
#' @param momentum The momentum for training. Default = 0.8. (Not Implemented!)
#' @param rand_seed Random seed. Default = 1234.
#' @keywords MLP
#' @export
#' @examples
#' ## Simple example
#' train_X <- array(c(1, 1, 1, 0, 0, 0,
#'                    1, 0, 1, 0, 0, 0,
#'                    1, 1, 1, 0, 0, 0,
#'                    0, 0, 1, 1, 1, 0,
#'                    0, 0, 1, 0, 1, 0,
#'                    0, 0, 1, 1, 1, 0),
#'                    c(6, 6));
#' train_Y <- array(c(1, 0,
#'                    1, 0,
#'                    1, 1,
#'                    0, 0,
#'                    0, 1,
#'                    0, 1), c(6, 2));
#'
#' batch_size <- 2;
#'
#' layers <- list(
#'    c("LINE", 6, 2, 0.1, batch_size),
#'    c("RELU", 2, batch_size),
#'    c("SOFT", 2, batch_size)
#' );
#' mlp1 <- train.mlp(train_X, train_Y, layers);
#'
#' ## The MNIST example
#' data(mnist)
#'
#' train <- mnist$train;
#' training_x <- train$x / 255;
#' training_y <- binarization(train$y);
#'
#' test <- mnist$test;
#' testing_x <- test$x / 255;
#' testing_y <- binarization(test$y);
#'
#' dataset <- training_x[1:1000,, drop=FALSE];
#' targets <- training_y[1:1000,, drop=FALSE];
#'
#' newdata <- testing_x[1:1000,, drop=FALSE];
#'
#' batch_size <- 10;
#' training_epochs <- 10;
#' learning_rate <- 1e-3;
#' momentum <- 0.8;
#' rand_seed <- 1234;
#'
#' layers <- list(
#'              c("LINE", 784, 64, 0.1, batch_size),
#'              c("RELV", batch_size),
#'              c("LINE", 64, 10, 0.1, batch_size),
#'              c("SOFT", 10, batch_size)
#' );
#'
#' mnist_mlp <- train.mlp(dataset, targets, layers, batch_size, training_epochs,
#'                        learning_rate, momentum, rand_seed);
train.mlp <- function (dataset, targets, layers,  batch_size = 10,
			training_epochs = 10, learning_rate = 1e-3,
			momentum = 0.8, rand_seed = 1234)
{
	if (is.null(dataset) || is.null(targets) || is.null(layers))
	{
		message("The input dataset, targets or layers are NULL");
		return(NULL);
	}
	
	if (is.null(batch_size) || is.null(training_epochs)
	|| is.null(learning_rate) || is.null(momentum) | is.null(rand_seed))
	{
		message("Some mandatory parameters are NULL");
		return(NULL);
	}
	
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

	retval <- .Call("_C_MLP_train", as.matrix(dataset), as.matrix(targets),
		as.list(layers), as.integer(length(layers)), as.integer(batch_size),
		as.integer(training_epochs), as.double(learning_rate), as.double(momentum),
		as.integer(rand_seed), PACKAGE = "rcnn");

	class(retval) <- c("mlp", class(retval));

	retval;
}

#' Predicting using a MultiLayer Perceptron Network Function
#'
#' This function predicts a dataset using a trained MLP. Admits as parameters
#' the testing dataset, and a trained MLP. Returns a matrix of predicted
#' outputs.
#' @param mlp A trained MLP.
#' @param newdata A matrix with data, one example per row.
#' @keywords MLP
#' @export
#' @examples
#' test_X <- array(c(1, 1, 1, 0, 0, 0,
#'                   1, 0, 1, 0, 0, 0,
#'                   1, 1, 1, 0, 0, 0,
#'                   0, 0, 1, 1, 1, 0,
#'                   0, 0, 1, 0, 1, 0,
#'                   0, 0, 1, 1, 1, 0),
#'                   c(6, 6));
#'
#' results <- predict(mlp1, test_X);
#'
#' ## The MNIST example
#' data(mnist)
#'
#' train <- mnist$train;
#' training_x <- train$x / 255;
#' training_y <- binarization(train$y);
#'
#' test <- mnist$test;
#' testing_x <- test$x / 255;
#' testing_y <- binarization(test$y);
#'
#' dataset <- training_x[1:1000,, drop=FALSE];
#' targets <- training_y[1:1000,, drop=FALSE];
#'
#' newdata <- testing_x[1:1000,, drop=FALSE];
#'
#' batch_size <- 10;
#' training_epochs <- 10;
#' learning_rate <- 1e-3;
#' momentum <- 0.8;
#' rand_seed <- 1234;
#'
#' layers <- list(
#'              c("LINE", 784, 64, 0.1, batch_size),
#'              c("RELV", batch_size),
#'              c("LINE", 64, 10, 0.1, batch_size),
#'              c("SOFT", 10, batch_size)
#' );
#'
#' mnist_mlp <- train.mlp(dataset, targets, layers, batch_size, training_epochs,
#'                        learning_rate, momentum, rand_seed);
#'
#' prediction <- predict(mnist_mlp, newdata);
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

	.Call("_C_MLP_predict", as.matrix(newdata), as.list(mlp$layers),
		as.integer(length(mlp$layers)), PACKAGE = "rcnn");
}

