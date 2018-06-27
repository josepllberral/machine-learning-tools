###############################################################################
# FEED-FORWARD ARTIFICIAL NEURAL NETWORK in R                                 #
###############################################################################

## @author Josep Ll. Berral (Barcelona Supercomputing Center)

## @date 24 February 2017

## References:
## * Approach based on Peng Zhao's "R for Deep Learning":
##   http://www.parallelr.com/r-deep-neural-network-from-scratch/

## Mocap data:
## * Iris Dataset: R.A. Fisher, 1935

###############################################################################
# GENERIC FUNCTIONS                                                           #
###############################################################################

## Function to produce Normal Samples
sample_normal <- function(dims, mean = 0, sd = 1)
{
	array(rnorm(n = prod(dims), mean = mean, sd = sd), dims);
}

## Function to produce the SoftMax
softmax <- function(score)
{
	score.exp <- exp(score);
	sweep(score.exp, 1, rowSums(score.exp), '/');
}

###############################################################################
# FFANN FUNCTIONS                                                             #
###############################################################################

## Feed-Forward Artificial Neural Network (FFANN). Constructor
create_ffann <- function(n_visible = 1, n_hidden = 6, n_output = 2, rand_seed = 1234)
{
	set.seed(rand_seed);

	W1 <- 0.01 * sample_normal(c(n_visible, n_hidden));
	b1 <- array(0, c(1, n_hidden));

	W2 <- 0.01 * sample_normal(c(n_hidden, n_output));
	b2 <- array(0, c(1, n_output));

	velocity <- list(W1 = array(0, dim(W1)), W2 = array(0, dim(W2)),
		b1 = rep(0, length(b1)), b2 = rep(0, length(b2)));

	list(n_visible = n_visible, n_hidden = n_hidden, n_output = n_output,
		W1 = W1, W2 = W2, b1 = b1, b2 = b2, velocity = velocity);
}

## This function computes the input through the hidden layers, then throug softmax
move_forward_ffann <- function (ffann, input)
{
	hidden.layer <- pmax(sweep(input %*% ffann$W1, 2, ffann$b1, '+'), 0);
	score <- sweep(hidden.layer %*% ffann$W2, 2, ffann$b2, '+');

	probs <- softmax(score);

	list(hidden.layer = hidden.layer, probs = probs);
}

## This function computes the output backwards the hidden layers
move_backward_ffann <- function (ffann, input, y.idx, hidden.layer, probs)
{
	dscores <- probs;
	dscores[y.idx] <- dscores[y.idx] - 1;
	dscores <- dscores / nrow(input);

	Delta_W2 <- t(hidden.layer) %*% dscores;
	Delta_b2 <- colSums(dscores);

	dhidden <- dscores %*% t(ffann$W2);
	dhidden[hidden.layer <= 0] <- 0;

	Delta_W1 <- t(input) %*% dhidden;
	Delta_b1 <- colSums(dhidden);

	list(Delta_W1 = Delta_W1, Delta_W2 = Delta_W2, Delta_b1 = Delta_b1, Delta_b2 = Delta_b2);
}

## This functions implements an iteration over back-propagation
##	param input: matrix input from batch data (n_obs x features)
##	param y.idx: vector with indices towards the output label
##	param lr: learning rate used to train the FFANN
##	param reg: regularization rate to train the FFANN
##	We assume sigma = 1 when computing deltas
get_cost_updates_ffann <- function(ffann, input, y.idx, lr, reg)
{
	# compute positive phase (forward)
	ph <- move_forward_ffann(ffann, input);

	# compute negative phase (backwards)
	nh <- move_backward_ffann(ffann, input, y.idx, ph[["hidden.layer"]], ph[["probs"]]);

	# compute the loss
	data.loss  <- sum(-log(ph$probs[y.idx])) / nrow(input);
	reg.loss   <- 0.5 * reg * (sum(ffann$W1 * ffann$W1) + sum(ffann$W2 * ffann$W2));
	cost <- data.loss + reg.loss;

	# update the network
	ffann$W1 <- ffann$W1 - lr * (nh$Delta_W1  + reg * ffann$W1);
	ffann$b1 <- ffann$b1 - lr * nh$Delta_b1;

	ffann$W2 <- ffann$W2 - lr * (nh$Delta_W2 + reg * ffann$W2)
	ffann$b2 <- ffann$b2 - lr * nh$Delta_b2;

	list(ffann = ffann, cost = cost);
}


###############################################################################
# HOW TO TRAIN YOUR FFANN                                                     #
###############################################################################

# Train: build and train a 2-layers neural network 

## Function to train the FFANN
##	param traindata: training dataset
##	param testdata: testing dataset
##	param varin: input variables (index or names)
##	param varout: output variable
##	param hidden: number of neurons in hidden layer
##	param maxit: max iteration steps
##	param abstol: delta loss 
##	param lr: learning rate
##	param reg: regularization rate
##	param display: show results every 'display' step
train_ffann <- function(traindata, varin, varout, testdata = NULL,
		n_hidden = c(6), maxit = 2000, abstol = 1e-2, lr = 1e-2,
		reg = 1e-3, display = 100, rand_seed = 1234)
{
	set.seed(rand_seed);
 
	# Training Data
	batchdata <- unname(data.matrix(traindata[,varin]));
	labels <- traindata[,varout];
	if (is.factor(labels)) { labels <- as.integer(labels); }

	# create index for both row and col
	y.set <- sort(unique(labels));
	y.idx <- cbind(1:nrow(batchdata), match(labels, y.set));
 
	# number of input features and categories for classification
	n_dim <- ncol(batchdata);
	n_classes <- length(unique(labels));

	# create the FFANN object
	ffann <- create_ffann(n_visible = n_dim, n_hidden = n_hidden,
			  n_output = n_classes, rand_seed = rand_seed
	);

	# training the FFANN
	i <- 1;
	cost <- 9e+15;
	while(i <= maxit && cost > abstol)
	{
		# training: get cost and update model
		aux <- get_cost_updates_ffann(ffann, batchdata, y.idx, lr, reg);

		# update network and loss
		ffann <- aux$ffann;
		cost <- aux$cost;

		# display results and update model
		if( i %% display == 0 )
		{
			accuracy <- NULL;
			if(!is.null(testdata))
			{
				labs <- predict_ffann(ffann, testdata[,-varout]);
				accuracy <- mean(as.integer(testdata[,varout]) == y.set[labs]);
			}
			message(i, " ", cost, " ",accuracy);
		}

 		i <- i + 1;
	}

	return(ffann);
}

###############################################################################
# PREDICTING VALUES                                                           #
###############################################################################

## Produce a prediction for new data.
##	param data : data matrix of (observations x n_visible)
##
## Feed Forward:
## - neurons : Rectified Linear
## - Loss Function: softmax
## - select max possiblity
predict_ffann <- function(ffann, data)
{
	new.data <- data.matrix(data);

	hidden.layer <- pmax(sweep(new.data %*% ffann$W1 ,2, ffann$b1, '+'), 0);
	score <- sweep(hidden.layer %*% ffann$W2, 2, ffann$b2, '+');
	probs <- softmax(score);

	labels.predicted <- max.col(probs);
}

###############################################################################
# IRIS EXAMPLE                                                                #
###############################################################################

## Main Function - Program Entry Point
main <- function()
{
	#options(width=as.integer(Sys.getenv("COLUMNS")));

	set.seed(1234);
	 
	# split data into test/train
	samp <- c(sample(1:50,25), sample(51:100,25), sample(101:150,25));
	 
	# train model
	ir.model <- train_ffann(varin = 1:4, varout = 5,
				traindata = iris[samp,],
				testdata = iris[-samp,],
				n_hidden = 6,
				maxit = 2000,
				display = 50
	);
	 
	# prediction
	labels.ffann <- predict_ffann(ir.model, iris[-samp, -5]);
	 
	# show results
	print(table(iris[-samp,5], labels.ffann));
	print(mean(as.integer(iris[-samp, 5]) == labels.ffann));
}

