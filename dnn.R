###############################################################################
# DEEP FEED-FORWARD ARTIFICIAL NEURAL NETWORK in R                            #
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
# DNN FUNCTIONS                                                               #
###############################################################################

## Deep Neural Network (DNN). Constructor
create_dnn <- function(n_visible = 1, n_hidden = c(6,5), n_output = 2, rand_seed = 1234)
{
	set.seed(rand_seed);

	layers <- length(n_hidden);

	W.list <- list();
	b.list <- list();

	prev_units <- n_visible;
	for (i in 1:layers)
	{
		units <- n_hidden[i];

		W.list[[i]] <- 0.01 * sample_normal(c(prev_units, units));
		b.list[[i]] <- array(0, c(1, units));

		prev_units <- units;
	}
	W.list[[layers + 1]] <- 0.01 * sample_normal(c(prev_units, n_output));
	b.list[[layers + 1]] <- array(0, c(1, n_output));

	list(n_visible = n_visible, n_hidden = n_hidden, n_output = n_output,
		W = W.list, b = b.list, layers = layers);
}

## This function computes the input through the hidden layers, then throug softmax
move_forward_dnn <- function (dnn, input)
{
	hl.list <- list();

	data.input <- input;
	for (i in 1:dnn$layers)
	{
		data.input <- pmax(sweep(data.input %*% dnn$W[[i]], 2, dnn$b[[i]], '+'), 0);		
		hl.list[[i]] <- data.input;
	}
	score <- sweep(data.input %*% dnn$W[[dnn$layers + 1]], 2, dnn$b[[dnn$layers + 1]], '+');

	probs <- softmax(score);

	list(hidden.layers = hl.list, probs = probs);
}

## This function computes the output backwards the hidden layers
move_backward_dnn <- function (dnn, input, y.idx, hidden.layers, probs)
{
	Delta_W <- list();
	Delta_b <- list();

	dscores <- probs;
	dscores[y.idx] <- dscores[y.idx] - 1;
	dhidden <- dscores / nrow(input);

	for (i in dnn$layers:1)
	{
		hidden.layer <- hidden.layers[[i]];

		Delta_W[[i + 1]] <- t(hidden.layer) %*% dhidden;
		Delta_b[[i + 1]] <- colSums(dhidden);

		dhidden <- dhidden %*% t(dnn$W[[i + 1]]);
		dhidden[hidden.layer <= 0] <- 0;
	}

	Delta_W[[1]] <- t(input) %*% dhidden;
	Delta_b[[1]] <- colSums(dhidden);

	list(Delta_W = Delta_W, Delta_b = Delta_b);
}

## This functions implements an iteration over back-propagation
##	param input: matrix input from batch data (n_obs x features)
##	param y.idx: vector with indices towards the output label
##	param lr: learning rate used to train the DNN
##	param reg: regularization rate to train the DNN
##	We assume sigma = 1 when computing deltas
get_cost_updates_dnn <- function(dnn, input, y.idx, lr, reg)
{
	# compute positive phase (forward)
	ph <- move_forward_dnn(dnn, input);

	# compute negative phase (backwards)
	nh <- move_backward_dnn(dnn, input, y.idx, ph[["hidden.layers"]], ph[["probs"]]);

	# compute the loss
	data.loss  <- sum(-log(ph$probs[y.idx])) / nrow(input);
	reg.loss   <- reg * mean(sapply(1:(dnn$layers + 1), function(x) sum(dnn$W[[x]] * dnn$W[[x]])));
	cost <- data.loss + reg.loss;

	# update the network
	for (i in 1:(dnn$layers+1))
	{
		dnn$W[[i]] <- dnn$W[[i]] - lr * (nh$Delta_W[[i]]  + reg * dnn$W[[i]]);
		dnn$b[[i]] <- dnn$b[[i]] - lr * nh$Delta_b[[i]];
	}

	list(dnn = dnn, cost = cost);
}


###############################################################################
# HOW TO TRAIN YOUR DNN                                                       #
###############################################################################

# Train: build and train a N-layers neural network 

## Function to train the DNN
##	param traindata: training dataset
##	param testdata: testing dataset
##	param varin: input variables (index or names)
##	param varout: output variable
##	param hidden: vector of neurons per hidden layer
##	param maxit: max iteration steps
##	param abstol: delta loss 
##	param lr: learning rate
##	param reg: regularization rate
##	param display: show results every 'display' step
train_dnn <- function(traindata, varin, varout, testdata = NULL,
		n_hidden = c(6, 5), maxit = 2000, abstol = 1e-2, lr = 1e-2,
		reg = 1e-3, display = 100, rand_seed = 1234)
{
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

	# create the DNN object
	dnn <- create_dnn(n_visible = n_dim, n_hidden = n_hidden,
			  n_output = n_classes, rand_seed = rand_seed
	);

	# training the DNN
	i <- 1;
	cost <- 9e+15;
	while(i <= maxit && cost > abstol)
	{
		# training: get cost and update model
		aux <- get_cost_updates_dnn(dnn, batchdata, y.idx, lr, reg);

		# update network and loss
		dnn <- aux$dnn;
		cost <- aux$cost;

		# display results and update model
		if( i %% display == 0 )
		{
			accuracy <- NULL;
			if(!is.null(testdata))
			{
				labs <- predict_dnn(dnn, testdata[,-varout]);
				accuracy <- mean(as.integer(testdata[,varout]) == y.set[labs]);
			}
			message(i, " ", cost, " ",accuracy);
		}

 		i <- i + 1;
	}

	return(dnn);
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
predict_dnn <- function(dnn, data)
{
	data.input <- data.matrix(data);
	for (i in 1:dnn$layers)
	{
		data.input <- pmax(sweep(data.input %*% dnn$W[[i]] ,2, dnn$b[[i]], '+'), 0);	
	}
	score <- sweep(data.input %*% dnn$W[[dnn$layers + 1]], 2, dnn$b[[dnn$layers + 1]], '+');
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
	ir.model <- train_dnn(varin = 1:4, varout = 5,
				traindata = iris[samp,],
				testdata = iris[-samp,],
				n_hidden = c(6,6),
				maxit = 12000,
				display = 500
	);
	 
	# prediction
	labels.dnn <- predict_dnn(ir.model, iris[-samp, -5]);
	 
	# show results
	print(table(iris[-samp,5], labels.dnn));
	print(mean(as.integer(iris[-samp, 5]) == labels.dnn));
}

