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
#' descriptor <- list(
#'	c('type' = "CONV", 'n_channels' = 1, 'n_filters' = 4, 'filter_size' = 5, 'scale' = 0.1, 'border_mode' = 'same'),
#'	c('type' = "POOL", 'n_channels' = 4, 'scale' = 0.1, 'win_size' = 3, 'stride' = 2),
#'	c('type' = "RELU", 'n_channels' = 4),
#'	c('type' = "CONV", 'n_channels' = 4, 'n_filters' = 16, 'filter_size' = 5, 'scale' = 0.1, 'border_mode' = 'same'),
#'	c('type' = "POOL", 'n_channels' = 16, 'scale' = 0.1, 'win_size' = 3, 'stride' = 2),
#'	c('type' = "RELU", 'n_channels' = 16),
#'	c('type' = "FLAT", 'n_channels' = 16),
#'	c('type' = "LINE", 'n_visible' = 784, 'n_hidden' = 64, 'scale' = 0.1),
#'	c('type' = "RELV"),
#'	c('type' = "LINE", 'n_visible' = 64, 'n_hidden' = 10, 'scale' = 0.1),
#'	c('type' = "SOFT", 'n_inputs' = 10)
#' );
#' is_valid <- check_layers (descriptor, dim(training_x), dim(training_y), batch_size = 10);
check_layers <- function (layers, dim_dataset, dim_target, batch_size)
{
	# Input Dimensions
	nrow <- dim_dataset[1];
	ncol <- dim_dataset[2];
	if (length(dim_dataset) == 4)
	{
		img_h <- dim_dataset[3];
		img_w <- dim_dataset[4];

		input_dims <- c(batch_size, dim_dataset[2:4]);
	} else if (length(dim_dataset) == 2)
	{
		input_dims <- c(batch_size, dim_dataset[2]);
	}

	nrow_y <- dim_target[1];
	ncol_y <- dim_target[2];

	# Check inputs vs outputs
	if (nrow != nrow_y)
	{
		message(paste("Error in Inputs. Dataset:", nrow, "Target:", nrow_y, sep = " "));
		message("Inputs and Output rows do not match");
		return (FALSE);
	}

	# Check batch_size vs number of samples
	if (batch_size > nrow)
	{
		message(paste("Error in Batch_size. Dataset:", nrow, " < Batch size:", batch_size, sep = " "));
		message("Batch size is larger than number of samples");
		return (FALSE);
	}

	nlayers <- length(layers);

	# Check Pipeline
	for (i in 1:nlayers)
	{
		laux <- layers[[i]];

		# Check for valid values
		idx <- which(names(laux) %in% c('type','border_mode'));
		pass <- all(!is.na(as.numeric(laux[-idx])));
		pass <- pass && all(as.numeric(laux[-idx]) > 0);
		if (!pass)
		{
			message(paste("Error in layer ", i, sep = ""));
			message("Incorrect input value (negative, character or zero...?)");
			return (FALSE);
		}

		# Check for Layers
		if (laux['type'] == "CONV")
		{
			# Check for Channels
			if (input_dims[2] != as.numeric(laux['n_channels']))
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current CONV input (channels) do not match previous LAYER output (channels)");
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
				return (FALSE);
			}
			input_dims[2] <- as.numeric(laux[3]);
		} else if (laux['type'] == "POOL")
		{
			# Check for Channels
			if (input_dims[2] != as.numeric(laux['n_channels']))
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current POOL input (channels) do not match previous LAYER output (channels)");
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
				return (FALSE);
			}
			out_h <- (input_dims[3] - as.numeric(laux['win_size']) + 2 * as.numeric(laux['win_size']) %/% 2) %/% as.numeric(laux['stride']) + 1;
			out_w <- (input_dims[4] - as.numeric(laux['win_size']) + 2 * as.numeric(laux['win_size']) %/% 2) %/% as.numeric(laux['stride']) + 1;
			input_dims[3:4] <- c(out_h, out_w);
		} else if (laux['type'] == "RELU")
		{
			# Check for Channels
			if (input_dims[2] != as.numeric(laux['n_channels']))
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current RELU input (channels) do not match previous LAYER output (channels)");
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
				return (FALSE);
			}
		} else if (laux['type'] == "FLAT")
		{
			# Check for Channels
			if (input_dims[2] != as.numeric(laux['n_channels']))
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current FLAT input (channels) do not match previous LAYER output (channels)");
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
				return (FALSE);
			}
			input_dims <- c(input_dims[1], prod(input_dims[-1]));
		} else if (laux['type'] == "LINE")
		{
			# Check for Visible units
			if (input_dims[2] != as.numeric(laux['n_visible']))
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current LINE input (visible) do not match previous LAYER output (visible)");
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
				return (FALSE);
			}
			input_dims[2] <- as.numeric(laux['n_hidden']);
		} else if (laux['type'] %in% c("SOFT", "SIGM", "TANH", "DIRE"))
		{
			# Check for Visible units
			if (input_dims[2] != as.numeric(laux['n_inputs']))
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current SOFT/SIGM/TANH/DIRE input (visible) do not match previous LAYER output (visible)");
				message(paste("Expected dimensions ", paste(input_dims, collapse = " "), sep = ""));
				return (FALSE);
			}
		} else if (laux['type'] %in% c("RELV"))
		{
			# Nothing to check here
			next
		} else
		{
			message(paste("Error in layer ", i, sep = ""));
			message(paste("Unknown type of layer ", laux['type']), sep = "");
			return (FALSE);
		}
	}

	# Check Last Layer
	if (!layers[[nlayers]]['type'] %in% c("SOFT","SIGM","LINE","DIRE","TANH"))
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

# Private function to complete layers from layer descriptor to better format
# for the C library
compose_layers <- function(descriptor, batch_size)
{
	layers <- list();
	
	for (i in 1:length(descriptor))
	{
		aux <- descriptor[[i]];
		if (aux['type'] == "CONV") {
			if (aux['border_mode'] == 'same') { bm <- 2 } else { bm <- 1 } # FIXME
			l <- c("CONV", aux['n_channels'], aux['n_filters'], aux['filter_size'], aux['scale'], bm);
		} else if (aux['type'] == "POOL") {
			l <- c("POOL", aux['n_channels'], aux['scale'], aux['win_size'], aux['stride']);
		} else if (aux['type'] == "RELU") {
			l <- c("RELU", aux['n_channels']);
		} else if (aux[1] == "FLAT") {
			l <- c("FLAT", aux['n_channels']);
		} else if (aux[1] == "LINE") {
			l <- c("LINE", aux['n_visible'], aux['n_hidden'], aux['scale']);
		} else if (aux[1] == "SOFT") {
			l <- c("SOFT", aux['n_inputs']);
		} else if (aux[1] == "SIGM") {
			l <- c("SIGM", aux['n_inputs']);
		} else if (aux[1] == "TANH") {
			l <- c("TANH", aux['n_inputs']);
		} else if (aux[1] == "DIRE") {
			l <- c("DIRE", aux['n_inputs']);
		} else if (aux['type'] == "RELV") {
			l <- c("RELV");
		} else {
			message("Error in Network Descriptor");
			message(paste("Layer", i, "has incorrect parameters"), sep = " ");
			return (NULL);
		}
		layers[[i]] <- l;
	}
	layers;
}

#' Training a Convolutional Neural Network or MultiLayer Perceptron Function
#'
#' This function trains a CNN or MLP. Admits as parameters the training dataset,
#' the matrix of outputs, and a descriptor of the network including all the
#' layers and their properties. Returns a CNN in list form, including all the
#' trained layers, if input is a 4D dataset (samples x image with depth), or an
#' MLP in list form, including all the trained layers, if input is a 2D dataset
#' (samples x features).
#' Possible layers are:
#' \itemize{
#'   \item CONV: Convolutional Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of channels of the input image
#'     \item Number of filters to be returned
#'     \item Size of convolutional filters
#'     \item Scale for initialization weights
#'     \item Border Mode (1 = valid, 2 = same, 3 = full)
#'   }
#'   \item POOL: Pooling Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of channels of the input image.
#'     \item Scale for initialization weights
#'     \item Window size
#'     \item Stride
#'   }
#'   \item RELU: Rectified Linear. Requires, in the following order:
#'   \enumerate{
#'     \item Number of channels of the input image.
#'   }
#'   \item FLAT: Flattening Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of channels of the input image.
#'   }
#'   \item LINE: Linear Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'     \item Number of hidden units
#'     \item Scale for initialization weights
#'   }
#'   \item RELV: Rectified Linear (for flattened batches). Does not require
#'   parameters
#'   \item SOFT: SoftMax Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'   }
#'   \item SIGM: Sigmoid Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'   }
#'   \item TANH: Hyperbolic Tangent Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'   }
#'   \item DIRE: Direct (buffer) Layer. Requires, in the following order:
#'   \enumerate{
#'     \item Number of visible units
#'   }
#' }
#' The list of layers and dimensionsis checked before the process starts.
#' Convolutional, Pooling, Rectifiers and Flattening layers can only be used
#' in CNNs. Other layers can be applied to CNNs and MLPs.
#' @param dataset A matrix with data, one example per row.
#' @param targets A matrix with output labels, one set of targets per row.
#' @param batch_size Number of examples per training mini-batch. Default = 1.
#' @param training_epochs Number of training epochs. Default = 1000.
#' @param learning_rate The learning rate for training. Default = 0.01.
#' @param momentum The momentum for training. Default = 0.8. (Not Implemented Yet!)
#' @param rand_seed Random seed. Default = 1234.
#' @keywords CNN MLP
#' @export
#' @examples
#' ## Simple example with CNN
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
#' layers <- list(
#'	c('type' = "CONV", 'n_channels' = 1, 'n_filters' = 4, 'filter_size' = 5, 'scale' = 0.1, 'border_mode' = 'same'),
#'	c('type' = "POOL", 'n_channels' = 4, 'scale' = 0.1, 'win_size' = 3, 'stride' = 2),
#'	c('type' = "RELU", 'n_channels' = 4),
#'	c('type' = "FLAT", 'n_channels' = 4),
#'	c('type' = "LINE", 'n_visible' = 8, 'n_hidden' = 2, 'scale' = 0.1),
#'	c('type' = "SOFT", 'n_inputs' = 2)
#' );
#'
#' cnn1 <- train.cnn(train_X, train_Y, layers, batch_size = 2);
#'
#' test_X <- array(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
#'                   0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0),
#'                   c(6, 1, 2, 3));
#' results <- predict(cnn1, test_X);
#'
#' ## The MNIST example with CNN
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
#' layers <- list(
#'	c('type' = "CONV", 'n_channels' = 1, 'n_filters' = 4, 'filter_size' = 5, 'scale' = 0.1, 'border_mode' = 'same'),
#'	c('type' = "POOL", 'n_channels' = 4, 'scale' = 0.1, 'win_size' = 3, 'stride' = 2),
#'	c('type' = "RELU", 'n_channels' = 4),
#'	c('type' = "CONV", 'n_channels' = 4, 'n_filters' = 16, 'filter_size' = 5, 'scale' = 0.1, 'border_mode' = 'same'),
#'	c('type' = "POOL", 'n_channels' = 16, 'scale' = 0.1, 'win_size' = 3, 'stride' = 2),
#'	c('type' = "RELU", 'n_channels' = 16),
#'	c('type' = "FLAT", 'n_channels' = 16),
#'	c('type' = "LINE", 'n_visible' = 784, 'n_hidden' = 64, 'scale' = 0.1),
#'	c('type' = "RELV"),
#'	c('type' = "LINE", 'n_visible' = 64, 'n_hidden' = 10, 'scale' = 0.1),
#'	c('type' = "SOFT", 'n_inputs' = 10)
#' );
#'
#' mnist_cnn <- train.cnn(dataset, targets, layers, batch_size = 10, training_epochs = 3,
#'                        learning_rate = 1e-3, momentum = 0.8, rand_seed = 1234);
#'
#' prediction <- predict(mnist_cnn, newdata);
#' 
#' ## Simple example with MLPs
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
#' layers <- list(
#'    c('type' = "LINE", 'n_visible' = 6, 'n_hidden' = 2, 'scale' = 0.1),
#'    c('type' = "RELV"),
#'    c('type' = "SOFT", 'n_inputs' = 2)
#' );
#' mlp1 <- train.cnn(train_X, train_Y, layers, batch_size = 2);
#'
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
#' ## The MNIST example with MLPs
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
#' layers <- list(
#'    c('type' = "LINE", 'n_visible' = 784, 'n_hidden' = 64, 'scale' = 0.1),
#'    c('type' = "RELV"),
#'    c('type' = "LINE", 'n_visible' = 64, 'n_hidden' = 10, 'scale' = 0.1),
#'    c('type' = "SOFT", 'n_inputs' = 10)
#' );
#'
#' mnist_mlp <- train.cnn(dataset, targets, layers, batch_size = 10, training_epochs = 10,
#'                        learning_rate = 1e-3, momentum = 0.8, rand_seed = 1234);
#'
#' prediction <- predict(mnist_mlp, newdata);
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

	if (!check_layers(layers, dim(dataset), dim(targets), batch_size))
	{
		message("Network does not match with data dimensions");
		return(NULL);
	}
	prep_layers <- compose_layers(layers, batch_size)

	if (length(dim(dataset)) == 4)
	{
		if ("integer" %in% class(dataset[1,1,1,1]))
		{
			message("Input matrix is Integer: Coercing to Numeric.");
			dataset <- 1.0 * dataset;
		}

		retval <- .Call("_C_CNN_train", as.array(dataset), as.matrix(targets),
			as.list(prep_layers), as.integer(length(prep_layers)), as.integer(batch_size),
			as.integer(training_epochs), as.double(learning_rate), as.double(momentum),
			as.integer(rand_seed), PACKAGE = "rcnn");

	} else if (length(dim(dataset)) == 2)
	{
		if ("integer" %in% class(dataset[1,1]))
		{
			message("Input matrix is Integer: Coercing to Numeric.");
			dataset <- t(apply(dataset, 1, as.numeric));
		}	
		
		# Check that all layers are Matricial
		for (i in 1:length(layers))
			if (!(layers[[i]][1] %in% c("LINE","RELV","SOFT","SIGM","TANH","DIRE")))
			{
				message(paste("Layers contain non-matricial layer", layers[[i]][1], "at", i, sep = " "));
				return(NULL);
			}

		retval <- .Call("_C_MLP_train", as.matrix(dataset), as.matrix(targets),
			as.list(prep_layers), as.integer(length(prep_layers)), as.integer(batch_size),
			as.integer(training_epochs), as.double(learning_rate), as.double(momentum),
			as.integer(rand_seed), PACKAGE = "rcnn");		
	} else
	{
		message("Error on Input dimensions: Must be a (Samples x Features) 2D matrix or a (Samples x Image) 4D array");
		return(NULL);
	}
	
	class(retval) <- c("cnn", class(retval));
	retval;
}

#' Predicting using a Convolutional Network or MultiLayer Perceptron Function
#'
#' This function predicts a dataset using a trained CNN. Admits as parameters
#' the testing dataset, and a trained CNN. Returns a matrix of predicted
#' outputs.
#' @param cnn A trained CNN.
#' @param mlp A trained MLP.
#' @param newdata A matrix with data, one example per row.
#' @keywords CNN MLP
#' @export
#' @examples
#' ## Simple example with CNNs
#' test_X <- array(c(1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
#'                   0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0),
#'                   c(6, 1, 2, 3));
#' results <- predict(cnn1, test_X);
#'
#' ## The MNIST example with CNNs
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
#' prediction <- predict(mnist_cnn, newdata);
#'
#' ## Simple example with MLPs
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
#' ## The MNIST example with MLPs
#' data(mnist)
#'
#' test <- mnist$test;
#' testing_x <- test$x / 255;
#' testing_y <- binarization(test$y);
#'
#' newdata <- testing_x[1:1000,, drop=FALSE];
#'
#' prediction <- predict(mnist_mlp, newdata);
predict.cnn <- function (cnn, newdata)
{
	if (!"cnn" %in% class(cnn))
	{
		message("Input object is not a CNN nor a MLP");
		return(NULL);
	}
		
	if (length(dim(newdata)) == 4)
	{	
		if ("integer" %in% class(newdata[1,1,1,1]))
		{
			message("Input matrix is Integer: Coercing to Numeric.");
			newdata <- 1.0 * newdata;
		}

		retval <- .Call("_C_CNN_predict", as.array(newdata), as.list(cnn$layers),
			  as.integer(length(cnn$layers)), PACKAGE = "rcnn");
		
	} else if (length(dim(newdata)) == 2)
	{
		if ("integer" %in% class(newdata[1,1]))
		{
			message("Input matrix is Integer: Coercing to Numeric.");
			newdata <- t(apply(newdata, 1, as.numeric));
		}

		retval <- .Call("_C_MLP_predict", as.matrix(newdata), as.list(cnn$layers),
			  as.integer(length(cnn$layers)), PACKAGE = "rcnn");
	} else
	{
		message("Error on Input dimensions: Must be a a (Samples x Features) 2D matrix or a (Samples x Image) 4D array");
		return(NULL);
	}

	list(score = retval, class = max.col(retval));
}
