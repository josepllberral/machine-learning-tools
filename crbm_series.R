###############################################################################
# CONDITIONAL RESTRICTED BOLTZMANN MACHINES in R for SERIES                   #
###############################################################################

## @author Josep Ll. Berral + Alberto Gutiérrez Torre
##         (Barcelona Supercomputing Center)

## This file extends the CRBMs, by training it from N series, instead of a
## single one. Each epoch iterates over the N series, splitting each in
## mini_batches for training.

source("crbm.R");

###############################################################################
# HOW TO TRAIN YOUR CRBM (SERIES)                                             #
###############################################################################

## Function to train the CRBM from N series
##	param learning_rate: learning rate used for training the CRBM
##	param training_epochs: number of epochs used for training
##	param dataset: loaded dataset <batchdata, seqlen, data_mean, data_std> for Motion
##	param batch_size: size of a batch used to train the CRBM
train_series_crbm <- function (dataset, learning_rate = 1e-4, momentum = 0.5, training_epochs = 300,
                               batch_size = 100, n_hidden = 100, delay = 6, n_gibbs = 1)
{
	set.seed(123);

	# get dimensions (e.g. from 1st serie)
	n_dim <- ncol(dataset$batchdata[[1]]);
	n_series <- length(dataset$batchdata);

	# construct the CRBM object
	crbm <- create_crbm(n_visible = n_dim, n_hidden = n_hidden, delay = delay, rand_seed = 123);
	crbm[["momentum"]] <- momentum;
	crbm[["training_epochs"]] <- training_epochs;
	crbm[["learning_rate"]] <- learning_rate;
	crbm[["batch_size"]] <- batch_size;

	# construct indexes for treating data
	list_permindex <- list_n_train_batches <- list();
	for (i in 1:n_series)
	{
		# valid starting indices
		batchdataindex <- NULL;
		last <- 1;
		for (s in dataset$seqlen[[i]])
		{
			batchdataindex <- c(batchdataindex, (last + delay):(last + s - 1));
			last <- last + s;
		}
		list_permindex[[i]] <- batchdataindex[sample(1:length(batchdataindex),length(batchdataindex))];

		# compute number of minibatches for training, validation and testing
		list_n_train_batches[[i]] <- ceiling(nrow(dataset$batchdata[[i]]) / batch_size);
	}

	start_time <- Sys.time();

	# go through the training epochs
	for (epoch in 1:training_epochs)
	{
		# go through the training set
		mean_cost <- NULL;

		# go through each series
		for (series in 1:n_series)
		{
			permindex <- list_permindex[[series]];
			n_train_batches <- list_n_train_batches[[series]]

			batchdata <- dataset$batchdata[[series]];
			seqlen <- dataset$seqlen[[series]];

			for (batch_index in 1:n_train_batches)
			{
				# linear index to the starting frames for this batch
				idx.aux.ini <- (((batch_index - 1) * batch_size) + 1);
				idx.aux.fin <- (batch_index * batch_size);
				if (idx.aux.fin > length(permindex)) break;
				data_idx <- permindex[idx.aux.ini:idx.aux.fin];

				# linear index to the frames at each delay tap
				hist_idx <- c(t(sapply(1:delay, function(x) data_idx - x)));
						       
				# update the CRBM parameters
				input <- batchdata[data_idx,];
				input_history <- t(array(c(t(batchdata[hist_idx,])), c(delay * n_dim, batch_size)));

				# get the cost and the gradient corresponding to one step of CD-k
				aux <- get_cost_updates_crbm(crbm, input, input_history, lr = learning_rate, momentum = momentum, k = n_gibbs);
				this_cost <- aux$recon;
				crbm <- aux$crbm;

				mean_cost <- c(mean_cost, this_cost);
			}
		}
		# if (epoch %% 10 == 0) message(paste('Training epoch ',epoch,', cost is ',mean(mean_cost, na.rm = TRUE),sep=""));
		crbm[["cost"]] <- mean(mean_cost, na.rm = TRUE);
	}

	end_time <- Sys.time();
	process_time <- end_time - start_time;
	# print(paste('Training took', process_time,sep=" "));
	crbm[["time"]] <- process_time;
	
	class(crbm) <- c("crbm", class(crbm));
	crbm;
}

###############################################################################
# PREDICTING VALUES (SERIES)                                                  #
###############################################################################

## Perform predict_crbm for a list of series
##      series : list of [batchdata, seqlen, means, stdevs]
##	n_gibbs : int, number of alternating Gibbs steps per iteration
predict_series_crbm <- function(crbm, series, n_gibbs = 30, n_threads = 1)
{
	generate_serie <- function(batchdata)
	{
		if (nrow(batchdata) < crbm$delay) return(NULL);

		samples.aux <- nrow(batchdata) - crbm$delay;

		data_idx <- c(crbm$delay + 1, crbm$delay + 1);
		orig_data <- batchdata[data_idx,];

		hist_idx <- c(sapply(data_idx, function(x) x - 1:crbm$delay));
		orig_history <- t(array(as.vector(t(batchdata[hist_idx,])), c(crbm$delay * crbm$n_visible, length(data_idx))));

		generated_series.aux <- predict_crbm(crbm, orig_data, orig_history, n_samples = samples.aux, n_gibbs = n_gibbs);

		oh.temp <- aperm(array(as.vector(orig_history), c(length(data_idx), crbm$n_visible, crbm$delay)),c(1,3,2));
		abind(oh.temp[,crbm$delay:1,], generated_series.aux, along = 2);
	}

	# Parallellization (if indicated)
	if (n_threads > 1)
	{
		library(parallel);
		cl <- makeCluster(n_threads, type='FORK');
		l <- parLapply(cl,series$batchdata, generate_serie);
		stopCluster(cl);
	} else {
		l <- lapply(series$batchdata, generate_serie);
	}

	l;
}

###############################################################################
# SIMULATION VALUES (SERIES)                                                  #
###############################################################################

## Simulates a step-by-step prediction using predict_crbm for an element of series
##      series : matrix of [time x dimensions] (a.k.a. batchdata)
simulate_serie_crbm <- function (crbm, series)
{
	batchdata <- series;

	activation.matrix <- NULL;
	activation.bin.matrix <- NULL;
	reconstruction.matrix <- NULL;

	if (crbm$delay + 1 > nrow(batchdata))
	{
#		message("Warning: Delay is longer than sequence");
		return(NULL);
	}

	for (i in (crbm$delay + 1):(nrow(batchdata)))
	{
		v.sample <- array(batchdata[i,,drop = FALSE], c(1,crbm$n_visible));
		v.history <- array(t(batchdata[(i-1):(i - crbm$delay),]), c(1,crbm$delay * crbm$n_visible));

		h.mean <- sigmoid_func((v.sample %*% crbm$W + v.history %*% crbm$B) %+% crbm$hbias);
		h.sample <- sample_bernoulli(h.mean);

		h.bins <- rep(0, length(h.mean));
		h.bins[h.mean > 0.5] <- 1;

		v.mean <- (h.mean %*% t(crbm$W) + v.history %*% crbm$A) %+% crbm$vbias;
		v.sample <- v.mean;

		activation.matrix <- rbind(activation.matrix, h.mean);
		activation.bin.matrix <- rbind(activation.bin.matrix, h.bins);
		reconstruction.matrix <- rbind(reconstruction.matrix, v.mean);
	}
	rownames(activation.matrix) <- NULL;
	rownames(activation.bin.matrix) <- NULL;
	rownames(reconstruction.matrix) <- NULL;

	list(batchdata = batchdata, activation.matrix = activation.matrix, activation.bin.matrix = activation.bin.matrix, reconstruction.matrix = reconstruction.matrix);
}

