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

dataset <- training_x[1:1000,,,, drop=FALSE];
targets <- training_y[1:1000,, drop=FALSE];

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

#check_layers(layers, dataset, targets, batch_size);

dyn.load("cnn.so")

#retval <- .Call("_C_CNN_train", as.array(dataset), as.matrix(targets), as.list(layers), as.integer(length(layers)), as.integer(batch_size), as.integer(training_epochs), as.double(learning_rate), as.double(momentum), as.integer(rand_seed));
retval <- train.cnn(dataset, targets, layers, 10, 1, 1e-3, 0.8, 1234);


###################################################################

## Function to check the basic shape of data between Layers
check_layers <- function (layers, dataset, target, batch_size)
{
	# Input Dimensions
	nrow <- dim(dataset)[1];
	ncol <- dim(dataset)[2];
	img_h <- dim(dataset)[3];
	img_w <- dim(dataset)[4];

	nrow_y <- dim(target)[1];
	ncol_y <- dim(target)[2];

	nlayers <- length(layers);

	# Check inputs vs outputs
	if (nrow != nrow_y)
	{
		message("Error in Inputs");
		message("Inputs and Output rows do not match");
		return (-1);
	}

	# Check Pipeline
	input_dims <- c(batch_size, dim(dataset)[2:4]);
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
		} else if (laux[1] %in% c("SOFT", "SIGM"))
		{
			# Check for Batch_size and Visible units
			if (all.equal(input_dims, as.numeric(laux[3:2])) != TRUE)
			{
				message(paste("Error in layer ", i, sep = ""));
				message("Current SOFT/SIGM input (batch_size, visible) do not match previous LAYER output (batch_size, visible)");
				return (FALSE);
			}
		}
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

	.Call("_C_CNN_predict", as.matrix(newdata), as.integer(rbm$n_visible),
		as.integer(rbm$n_hidden), as.matrix(rbm$W), as.numeric(rbm$hbias),
		as.numeric(rbm$vbias),
		PACKAGE = "rcnn"); # TODO
}

