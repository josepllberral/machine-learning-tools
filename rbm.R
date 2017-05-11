###############################################################################
# RESTRICTED BOLTZMANN MACHINES in R                                          #
###############################################################################

## @author Josep Ll. Berral (Barcelona Supercomputing Center)

# Inspired by the implementations from:
# * David Buchaca   : https://github.com/davidbp/connectionist
# * Andrew Landgraf : https://www.r-bloggers.com/restricted-boltzmann-machines-in-r/
# * Graham Taylor   : http://www.uoguelph.ca/~gwtaylor/

# Author of the snippet to load the MNIST example:
# * Brendan O'Connor https://gist.github.com/39760 - anyall.org

###############################################################################
# AUXILIAR FUNCTIONS                                                          #
###############################################################################

## Function to produce Normal Samples
sample_normal <- function(dims, mean = 0, sd = 1)
{
    array(rnorm(n = prod(dims), mean = mean, sd = sd), dims);
}

## Function to produce Bernoulli Samples
sample_bernoulli <- function(mat)
{
    dims <- dim(mat);
    array(rbinom(n = prod(dims), size = 1, prob = c(mat)), dims);
}

## Function to produce the Sigmoid
sigmoid_func <- function(mat)
{
    1 / (1 + exp(-mat));
}

## Operator to add dimension-wise vectors to matrices
`%+%` <- function(mat, vec)
{
    retval <- NULL;
    tryCatch(
        expr = { retval <- if (dim(mat)[1] == length(vec)) t(t(mat) + vec) else mat + vec; },
        warning = function(w) { print(paste("WARNING: ", w, sep = "")); },
        error = function(e) { print(paste("ERROR: Cannot sum mat and vec", e, sep = "\n")); }
    );
    retval;
}

###############################################################################
# RBM FUNCTIONS                                                               #
###############################################################################

## Restricted Boltzmann Machine (RBM). Constructor
create_rbm <- function (n_visible = 10, n_hidden = 100, W = NULL, hbias = NULL,
	vbias = NULL, batch_size = 100)
{
    if (is.null(W)) W <- 0.01 * sample_normal(c(n_visible, n_hidden));
    if (is.null(hbias)) hbias <- rep(0, n_hidden);
    if (is.null(vbias)) vbias <- rep(0, n_visible);
    
    velocity <- list(W = array(0, dim(W)), v = rep(0, length(vbias)), h = rep(0, length(hbias)));
    
    list(n_visible = n_visible, n_hidden = n_hidden, W = W, hbias = hbias,
	vbias = vbias, velocity = velocity, batch_size = batch_size);
}

### This function infers state of hidden units given visible units
visible_state_to_hidden_probabilities <- function(rbm, visible_state)
{
    h.mean <- sigmoid_func((visible_state %*% rbm$W) %+% rbm$hbias);
    h.sample <- sample_bernoulli(h.mean);

    list(mean = h.mean, sample = h.sample);
}

## This function infers state of visible units given hidden units
hidden_state_to_visible_probabilities <- function(rbm, hidden_state)
{
    v.mean <- (hidden_state %*% t(rbm$W)) %+% rbm$vbias;
    v.sample <- v.mean;

#    v.mean <- sigmoid_func((hidden_state %*% t(rbm$W)) %+% rbm$vbias);
#    v.sample <- sample_bernoulli(v.mean);

    list(mean = v.mean, sample = v.sample);
}

## This functions implements one step of CD-k
##  param input: matrix input from batch data (n_seq x n_vis)
##  param lr: learning rate used to train the RBM
##  param k: number of Gibbs steps to do in CD-k
##  param momentum: value for momentum coefficient on learning
cdk_rbm <- function(rbm, input, lr, k = 1, momentum = 0.1)
{  
    # compute positive phase (awake)
    ph <- visible_state_to_hidden_probabilities(rbm, input);

    # perform negative phase (asleep)
    nh <- ph;
    for (i in 1:k)
    {
        nv <- hidden_state_to_visible_probabilities(rbm, nh[["sample"]]);
        nh <- visible_state_to_hidden_probabilities(rbm, nv[["sample"]]);
    }

    # determine gradients on RBM parameters
    Delta_W <- t(input) %*% ph[["mean"]] - t(nv[["sample"]]) %*% nh[["mean"]];
    Delta_v <- colSums(input - nv[["sample"]]);
    Delta_h <- colSums(ph[["mean"]] - nh[["mean"]]);
    
    rbm$velocity[["W"]] <- rbm$velocity[["W"]] * momentum + lr * Delta_W / rbm$batch_size;
    rbm$velocity[["v"]] <- rbm$velocity[["v"]] * momentum + lr * Delta_v / rbm$batch_size;
    rbm$velocity[["h"]] <- rbm$velocity[["h"]] * momentum + lr * Delta_h / rbm$batch_size;
    
    # update weights
    rbm$W <- rbm$W + rbm$velocity[["W"]];
    rbm$vbias <- rbm$vbias + rbm$velocity[["v"]];
    rbm$hbias <- rbm$hbias + rbm$velocity[["h"]];

    # approximation to the reconstruction error: sum over dimensions, mean over cases
    list(rbm = rbm, recon = mean(colSums(`^`(input - nv[["mean"]],2))));
}

###############################################################################
# HOW TO TRAIN YOUR RBM                                                       #
###############################################################################

## Function to train the RBM
##  param dataset: loaded dataset (rows = examples, cols = features)
##  param learning_rate: learning rate used for training the RBM
##  param training_epochs: number of epochs used for training
##  param batch_size: size of a batch used to train the RBM
train_rbm <- function (dataset, batch_size = 100, n_hidden = 100,
	training_epochs = 300, learning_rate = 1e-4, momentum = 0.5,
	rand_seed = 1234)
{
    set.seed(rand_seed);

    n_train_batches <- ceiling(nrow(dataset) / batch_size);
    n_dim <- ncol(dataset);
      
    # shuffle indices
    permindex <- sample(1:nrow(dataset),nrow(dataset));

    # construct the RBM object
    rbm <- create_rbm(n_visible = n_dim, n_hidden = n_hidden, batch_size = batch_size);

    start_time <- Sys.time();
 
    # go through the training epochs and training set
    for (epoch in 1:training_epochs)
    {
	mean_cost <- NULL;
        for (batch_index in 1:n_train_batches)
        {
            idx.aux.ini <- (((batch_index - 1) * batch_size) + 1);
            idx.aux.fin <- idx.aux.ini + batch_size - 1;
            if (idx.aux.fin > length(permindex)) break;
            data_idx <- permindex[idx.aux.ini:idx.aux.fin];

            input <- dataset[data_idx,];
                       
            # get the cost and the gradient corresponding to one step of CD-k
            aux <- cdk_rbm(rbm, input, lr = learning_rate, momentum = momentum, k = 1);
            this_cost <- aux$recon;
            rbm <- aux$rbm;

            mean_cost <- c(mean_cost, this_cost);
        }
#	if (epoch %% 50 == 1)
		print(paste('Training epoch ',epoch,', cost is ',mean(mean_cost, na.rm = TRUE),sep=""));
    }

    end_time <- Sys.time();
    print(paste('Training took', (end_time - start_time), sep=" "));

    rbm;
}

###############################################################################
# PREDICTING VALUES                                                           #
###############################################################################

## Pass the current data through the RBM
##	dataset : data to be passed through the RBM
##	returns : list with activations and reconstructions
predict_rbm <- function(rbm, dataset)
{
    act.input <- sigmoid_func(dataset %*% rbm$W + rbm$hbias);
    rec.input <- act.input %*% t(rbm$W) + rbm$vbias;
    
    list(activations = act.input, reconstruction = rec.input);
}

###############################################################################
# EXPERIMENTS: THE MNIST EXAMPLE                                              #
###############################################################################

# Load the MNIST digit recognition dataset http://yann.lecun.com/exdb/mnist/
# into R. Assume you have all 4 files and gunzip'd them creates train$n,
# train$x, train$y and test$n, test$x, test$y e.g. train$x is a 60000 x 784
# matrix, each row is one digit (28x28) call: show_digit(train$x[5,]) to see a
# digit.
#
# Snippet authory: Brendan O'Connor https://gist.github.com/39760 - anyall.org
#
# Note: Put the MNIST data files in "./datasets/" folder

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

show_digit <- function(arr784, col=gray(12:1/12), ...)
{
    image(matrix(arr784, nrow=28)[,28:1], col=col, ...);
}

## Main Function - Program Entry Point
main <- function()
{
	aux <- load_mnist();
	training.num <- data.matrix(aux$train);
	testing.num <- data.matrix(aux$test);

	trainIndex <- sample(x=1:nrow(aux$train),size=nrow(aux$train) * 0.6);
	training <- aux$train[trainIndex,];
	cv <- aux$train[-trainIndex,];

	show_digit(as.matrix(training[5,2:785]));

	# Train an SVM to learn MNIST
	if (FALSE)
	{
	    opt.warn <- getOption("warn");
	    options(warn = -1);

	    # SVM. 95/94.
	    fit <- train(y ~ ., data = head(training, 1000), method = 'svmRadial', tuneGrid = data.frame(sigma=0.0107249, C=1))
	    results <- predict(fit, newdata = head(cv, 1000))
	    confusionMatrix(results, head(cv$y, 1000))

	    # Predict the digit.
	    predict(fit, newdata = training[5,])

	    # Check the actual answer for the digit.
	    training[5,1]

	    options(warn = opt.warn);
	}

	# Train an RBM to learn MNIST
	rbm <- train_rbm(
	    n_hidden = 30,
	    dataset = training.num[,2:785],
	    learning_rate = 1e-3,
	    training_epochs = 10,
	    batch_size = 10,
	    momentum = 0.5
	);

	# Plot the learned stuff
	library(ggplot2);
	library(reshape2);

	weights <- rbm$W;
	colnames(weights) <-  NULL;
	mw <- melt(weights);

	mw$Var3 <- floor((mw$Var2 - 1)/28) + 1;
	mw$Var2 <- (mw$Var2 - 1) %% 28 + 1;
	mw$Var3 <- 29 - mw$Var3;

	ggplot(data = mw) +
	geom_tile(aes(Var2, Var3, fill = value)) +
	facet_wrap(~Var1, nrow = 5) +
	scale_fill_continuous(low = 'white', high = 'black') +
	coord_equal() +
	labs(x = NULL, y = NULL, title = "Visualization of Weights") +
	theme(legend.position = "none")

	# Reconstruction of Values
	act.input <- sigmoid_func((rbm$W %*% t(training.num[,2:785])) + rbm$hbias);
	rec.input <- sigmoid_func((t(rbm$W) %*% act.input) + rbm$vbias);

	# Example of number in position 3
	show_digit(as.matrix(rec.input[,3]))
	show_digit(as.matrix(training.num[3,2:785]));
}
