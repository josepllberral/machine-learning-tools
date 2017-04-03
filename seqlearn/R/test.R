load.libs <- function()
{
	# This is for testing purposes
	system("cd src; R CMD SHLIB crbm.c rbm.c matrix_ops.c -lgsl -lgslcblas -o librbm.so");
	dyn.load("src/librbm.so");
	source("crbm.R");
	source("rbm.R");
}

## Example Library MNIST
load_mnist <- function()
{
	train <- data.frame();
	test <- data.frame();

	load_image_file <- function(filename)
	{
		ret = list();
		f = file(filename,'rb');
		readBin(f,'integer',n=1,size=4,endian='big');
		ret$n = readBin(f,'integer',n=1,size=4,endian='big');
		nrow = readBin(f,'integer',n=1,size=4,endian='big');
		ncol = readBin(f,'integer',n=1,size=4,endian='big');
		x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F);
		ret$x = matrix(x, ncol=nrow*ncol, byrow=T);
		close(f);
		ret;
	}

	load_label_file <- function(filename)
	{
		f = file(filename,'rb');
		readBin(f,'integer',n=1,size=4,endian='big');
		n = readBin(f,'integer',n=1,size=4,endian='big');
		y = readBin(f,'integer',n=n,size=1,signed=F);
		close(f);
		y;
	}

	train <- load_image_file('./datasets/train-images.idx3-ubyte');
	test <- load_image_file('./datasets/t10k-images.idx3-ubyte');

	train$x <- train$x / 255;
	test$x <- test$x / 255;

	train_y <- load_label_file('./datasets/train-labels.idx1-ubyte');
	test_y <- load_label_file('./datasets/t10k-labels.idx1-ubyte');


	train_y <- as.factor(train_y);
	test_y <- as.factor(test_y);

	inTrain <- data.frame(y = train_y, train$x);
	inTest <- data.frame(y = test_y, test$x);

	list(train = inTrain, test = inTest);
}

## Load Motion Example
load_motion <- function(filename)
{
	mat_dict <- readRDS(filename);		
	Motion <- mat_dict[['Motion']];

	n_seq <- length(Motion);
	
	# assume data is MIT format for now
	indx <- c(1:9, 14, 19:21, 26, 31:33, 38, 43:45, 50, 55:57, 61:63, 67:69, 73:75,
				79:81, 85:87, 91:93, 97:99, 103:105);

	# collapse sequences
	batchdata <- rbind(	Motion[[c(1,1)]][,indx],
				Motion[[c(2,1)]][,indx],
				Motion[[c(3,1)]][,indx]);
	data_mean <- colMeans(batchdata);
	data_std <- apply(batchdata, 2, sd);

	batchdata <- t((t(batchdata) - data_mean) / data_std);

	# get sequence lengths
	seqlen <- sapply(1:3, function(x) nrow(Motion[[c(x,1)]]));

	list(batchdata = batchdata, seqlen = seqlen, data_mean = data_mean, data_std = data_std);
}

## RBM Test function
testing.rbm <- function()
{
	aux <- load_mnist();
	training.num <- data.matrix(aux$train);
	testing.num <- data.matrix(aux$test);

	start.time <- Sys.time()
	rbm1 <- train.rbm(n_hidden = 30, dataset = training.num[2:785,], learning_rate = 1e-3, training_epochs = 10, batch_size = 10, momentum = 0.5 );
#	rbm1.r <- train_rbm(n_hidden = 30, dataset = training.num[2:785,], learning_rate = 1e-3, training_epochs = 10, batch_size = 10, momentum = 0.5 );
	end.time <- Sys.time()
	time.taken <- end.time - start.time
	time.taken

	res <- predict(rbm1, training.num[,2:785]);
	res;
}

## CRBM Test function
testing.crbm <- function()
{
	dataset <- load_motion('./datasets/motion.rds');

	start.time <- Sys.time()
	crbm <- train.crbm (dataset$batchdata, dataset$seqlen, batch_size = 100, n_hidden = 100, delay = 6, training_epochs = 200, learning_rate = 1e-3, momentum = 0.5, rand_seed = 1234);
#	crbm.r <- train_crbm (dataset, batch_size = 100, n_hidden = 100, delay = 6, training_epochs = 200, learning_rate = 1e-3, momentum = 0.5, rand_seed = 1234);
	end.time <- Sys.time()
	time.taken <- end.time - start.time
	time.taken

	res1 <- predict(crbm, dataset$batchdata);
	res2 <- forecast.crbm(crbm, dataset$batchdata[1:(crbm$delay+1),], 50, 30);

	list(predicted = res1, forecast = res2);
}


