
## Test using example Library MNIST
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

## Testing the CNN
testing.cnn <- function()
{
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

	newdata <- testing_x[1:1000,,,, drop=FALSE];

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
#		c("SIGM", 10, batch_size)
#		c("TANH", 10, batch_size)
#		c("DIRE", 10, batch_size)
	);

	check_layers(layers, dataset, targets, batch_size);

	# This is for testing purposes
#	dyn.load("cnn.so");
#	source("wrapper.R");

	start.time <- Sys.time()
	retval <- train.cnn(dataset, targets, layers, 10, 2, 1e-3, 0.8, 1234);
	end.time <- Sys.time()
	time.taken <- end.time - start.time
	time.taken

	prediction <- predict.cnn(retval, newdata);
}


## Testing the MLP
testing.mlp <- function()
{
	aux <- load_mnist();
	img_size <- c(28,28);

	train <- aux$train;
	training_x <- array(train$x, c(nrow(train$x), prod(img_size))) / 255;
	training_y <- binarization(train$y);

	test <- aux$test;
	testing_x <- array(test$x, c(nrow(test$x), prod(img_size))) / 255;
	testing_y <- binarization(test$y);

	dataset <- training_x[1:1000,, drop=FALSE];
	targets <- training_y[1:1000,, drop=FALSE];

	newdata <- testing_x[1:1000,, drop=FALSE];

	batch_size <- 10;
	training_epochs <- 1;
	learning_rate <- 1e-3;
	momentum <- 0.8;
	rand_seed <- 1234;

	layers <- list(
		c("LINE", 784, 64, 0.1, batch_size),
		c("RELV", batch_size),
		c("LINE", 64, 10, 0.1, batch_size),
		c("SOFT", 10, batch_size)
#		c("SIGM", 10, batch_size)
#		c("TANH", 10, batch_size)
#		c("DIRE", 10, batch_size)
	);

	# This is for testing purposes
#	dyn.load("cnn.so");
#	source("wrapper.R");

	check_layers(layers, dataset, targets, batch_size);

	start.time <- Sys.time()
	retval <- train.mlp(dataset, targets, layers, 10, 20, 1e-2, 0.8, 1234);
	end.time <- Sys.time()
	time.taken <- end.time - start.time
	time.taken

	prediction <- predict.mlp(retval, newdata);
}

