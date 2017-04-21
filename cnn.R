###############################################################################
# CONVOLUTIONAL NEURAL NETWORKS in R                                          #
###############################################################################

## @author Josep Ll. Berral (Barcelona Supercomputing Center)

## @date 3rd March 2017

## References:
## * Approach based on Lars Maaloee's:
##   https://github.com/davidbp/day2-Conv
## * Also from LeNet (deeplearning.net)
##   http://deeplearning.net/tutorial/lenet.html

## Mocap data:
## The MNIST digit recognition dataset http://yann.lecun.com/exdb/mnist/

## Libraries: We'll some speed-up for convolutions
library("Rcpp");
library("compiler")

## Functions for checking gradient correctness
source("grad_check.R");

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

## Performs the Convolution of mat (4D) using filter k (1,f,1,1)
cppFunction('
NumericMatrix conv2D(NumericMatrix mat, NumericMatrix k, String mode = "valid")
{
	int krow = k.nrow();
	int kcol = k.ncol();

	int krow_h = krow / 2;
	int kcol_h = kcol / 2;

	int mrow = mat.nrow();
	int mcol = mat.ncol();

	NumericMatrix out(mrow, mcol);

	for(int i = 0; i < mrow; ++i)
	{
		for(int j = 0; j < mcol; ++j)
		{
			double acc = 0;
			for(int m = 0; m < krow; ++m)
			{
				int mm = krow - 1 - m;
				int ii = i + (m - krow_h);

				if (ii >= 0 && ii < mrow)
					for(int n = 0; n < kcol; ++n)
					{
						int nn = kcol - 1 - n;
						int jj = j + (n - kcol_h);

						if (jj >= 0 && jj < mcol) acc += mat(ii,jj) * k(mm,nn);
					}
			}
			out(i,j) = acc;
		}
	}

	if (mode == "valid")
	{
		int cut_y = krow_h;
		int cut_x = kcol_h;

		int len_y = std::max(krow,mrow) - std::min(krow,mrow);
		int len_x = std::max(kcol,mcol) - std::min(kcol,mcol);
		out = out(Rcpp::Range(cut_y, cut_y + len_y), Rcpp::Range(cut_x, cut_x + len_x));
	}

	return out;
}
')

## Convolution - Old Version in "Native R"
#conv2D_legacy <- function(mat, k, mode = 'valid')
#{
#	out <- conv2D_sub(mat, k);
#	krow <- nrow(k);
#	kcol <- ncol(k);
#
#	krow_h <- krow %/% 2;
#	kcol_h <- kcol %/% 2;
#
#	mrow <- nrow(mat);
#	mcol <- ncol(mat);
#
#	out <- array(0, c(mrow,mcol));
#	for(i in 1:mrow)
#	{
#		for(j in 1:mcol)
#		{
#			acc <- 0;
#			for(m in 1:krow)
#			{
#				mm <- krow - m + 1;
#				ii <- i + m - krow_h - 1;
#				for(n in 1:kcol)
#				{
#					nn <- kcol - n + 1;
#					jj <- j + n - kcol_h - 1;
#
#					if( ii > 0 && ii <= mrow && jj > 0 && jj <= mcol)
#						acc <- acc + mat[ii,jj] * k[mm,nn];
#				}
#			}
#			out[i,j] <- acc;
#		}
#	}
#
#	if (mode == 'valid')
#	{
#		cut_y <- krow_h + 1;
#		cut_x <- kcol_h + 1;
#
#		len_y <- max(krow,mrow) - min(krow,mrow) + 1;
#		len_x <- max(kcol,mcol) - min(kcol,mcol) + 1;
#
#		out <- out[cut_y:(cut_y + len_y - 1),cut_x:(cut_x + len_x - 1)];
#	}
#
#	out;
#}

## Image Padding - Old Version in "Native R"
img_padding <- function(img, pad_y, pad_x)
{
	dims <- dim(img);
	imgs_pad <- array(0, c(dims[1] + 2 * pad_y, dims[2] + 2 * pad_x));

	aux <- cbind(img, array(0, c(nrow(img), pad_y)));
	aux <- cbind(array(0, c(nrow(aux), pad_y)), aux);
	aux <- rbind(aux, array(0, c(pad_x, ncol(aux))));
	aux <- rbind(array(0, c(pad_x, ncol(aux))), aux);
	aux;
}

binarization <- function(vec)
{
	result <- array(0, c(length(vec),length(unique(vec))));
	for (i in 1:length(vec)) result[i,vec[i]] <- 1;
	result;

#	t(sapply(1:length(a), function(x) { j <- rep(0,max(a)); j[a[x]] <- 1; j}))
}

###############################################################################
# CONVOLUTIONAL LAYERS                                                        #
###############################################################################

## Performs the convolution
##	param imgs: <batch_size, img_n_channels, img_height, img_width>
##	param filters: <n_filters, n_channels, win_height, win_width>
##	param padding: <padding_y, padding_x>
conv_bc01_orig <- function(imgs, filters, padding)
{
	# Compute shapes
	imgs.shape <- dim(imgs);
	batch_size <- imgs.shape[1];
	n_channels_img <- imgs.shape[2];
	img_h <- imgs.shape[3];
	img_w <- imgs.shape[4];

	filters.shape <- dim(filters);
	n_filters <- filters.shape[1];
	n_channels <- filters.shape[2];
	win_h <- filters.shape[3];
	win_w <- filters.shape[4];

	pad_y <- padding[1];
	pad_x <- padding[2];

	if (!(n_channels == n_channels_img))
	{
		warning('Mismatch in # of channels');
		return(NULL);
	}

	# Create output array
	out_h <- (img_h - win_h + 2 * pad_y) + 1;
	out_w <- (img_w - win_w + 2 * pad_x) + 1;
	out_shape <- c(batch_size, n_filters, out_h, out_w);
	out_1 <- array(0, out_shape);

	# Prepares padded image for convolution
	imgs_pad <- array(0, dim(imgs) + c(0, 0, 2*(pad_y), 2*(pad_x)));
	for (i in 1:dim(imgs)[1])
		for (j in 1:dim(imgs)[2])
			imgs_pad[i,j,,] <- img_padding(imgs[i,j,,], pad_y, pad_x);

	# Perform convolution
	for (b in 1:batch_size)
		for (f in 1:n_filters)
			for (c in 1:n_channels)	out_1[b,f,,] <- out_1[b,f,,] + conv2D(imgs_pad[b,c,,], filters[f,c,,]);

	return(out_1);
}
conv_bc01 <- cmpfun(conv_bc01_orig);

## Performs Forward Propagation
##	param x :	Array of shape (batch_size, n_channels, img_height, img_width)
##	return  :	Array of shape (batch_size, n_filters, out_height, out_width)
##	updates :	conv_layer
forward_conv_orig <- function(conv, x)
{      
	# Save "x" for back-propagation
	conv[["x"]] <- x;

        # Performs convolution
	y <- conv_bc01(x, conv$W, conv$padding);

	for (b in 1:dim(y)[1]) y[b,,,] <- y[b,,,] + conv$b[1,,1,1];

	list(layer = conv, y = y);
}
forward_conv <- cmpfun(forward_conv_orig);

## Performs Backward Propagation
##	param dy :	Array of shape (batch_size, n_filters, out_height, out_width)
##	return   :	Array of shape (batch_size, n_channels, img_height, img_width)
##	updates  :	conv_layer
backward_conv_orig <- function(conv, dy)
{
        # Flip weights
	w <- conv$W[,, (conv$w_shape[3]:1), (conv$w_shape[4]:1), drop = FALSE];

        # Transpose channel/filter dimensions of weights
	w <- aperm(w, c(2, 1, 3, 4));

	# Propagate gradients to x
        dx <- conv_bc01(dy, w, conv$padding);

	# Prepares padded image for convolution
	x_pad <- array(0, dim(conv$x) + c(0, 0, 2 * conv$padding));
	for (i in 1:dim(x_pad)[1])
		for (j in 1:dim(x_pad)[2])
			x_pad[i,j,,] <- img_padding(conv$x[i,j,,], conv$padding[1], conv$padding[2]);

        # Propagate gradients to weights and gradients to bias
	grad_W <- array(0, dim(conv$W));
	for (b in 1:dim(dy)[1])
		for (f in 1:dim(conv$W)[1])
			for (c in 1:dim(conv$W)[2])
				grad_W[f,c,,] <- grad_W[f,c,,] + conv2D(x_pad[b,c,,], dy[b,f,,]);

	conv[["grad_W"]] <- grad_W[,, (conv$w_shape[3]:1), (conv$w_shape[4]:1), drop = FALSE];
	conv[["grad_b"]] <- array(apply(dy, MARGIN = 2, sum), dim(conv$b));

	list(layer = conv, dx = dx);
}
backward_conv <- cmpfun(backward_conv_orig);

## Updates the Convolutional Layer
get_updates_conv_orig <- function(conv, lr)
{
	conv[["W"]] = conv$W - conv$grad_W * lr;
	conv[["b"]] = conv$b - conv$grad_b * lr;
	conv;
}
get_updates_conv <- cmpfun(get_updates_conv_orig);

## Get names of parameters and gradients (for testing functions)
pnames_conv <- function() { c("W","b"); }
gnames_conv <- function() { c("grad_W","grad_b"); }


## Returns a convolutional layer
create_conv <- function(n_channels, n_filters, filter_size, scale = 0.01, border_mode = 'same')
{
	dims <- c(n_filters, n_channels, filter_size, filter_size);

	W <- scale * sample_normal(dims);
	b <- array(0, c(1, n_filters, 1 ,1));

	if (border_mode == 'valid') padding <- 0;
	if (border_mode == 'same') padding <- filter_size %/% 2;
	if (border_mode == 'full') padding <- filter_size - 1;

	padding <- c(padding, padding);
	
	list(n_channels = n_channels, n_filters = n_filters, filter_size = filter_size,
		w_shape = dims,	W = W, b = b, padding = padding, pnames = pnames_conv, gnames = gnames_conv,
		forward = forward_conv, backward = backward_conv, get_updates = get_updates_conv);
}

## Driver for Convolutional Layer
main_conv <- function()
{
	batch_size <- 10;
	n_channels <- 1;
	img_shape <- c(5, 5);
	n_filters <- 2;
	filter_size <- 3;

	border_mode <- 'same';

	x <- sample_normal(c(batch_size, n_channels, img_shape));
	layer <- create_conv(n_channels, n_filters, filter_size, border_mode = border_mode);

	ok <- check_grad(layer, x);
	if (ok)	print('Gradient check passed') else print('Gradient check failed');
}

###############################################################################
# POOLING LAYERS                                                              #
###############################################################################

## Forwards a Pooling Matrix (4D) from a Convolutional Matrix (4D)
##	param x :	Array of shape (batch_size, n_channels, img_height, img_width)
##	returns :	Array of shape (batch_size, n_channels, out_height, out_width)
##	updates :	pool_layer
forward_pool_orig <- function(pool, imgs)
{
	# Compute shapes
	imgs.shape <- dim(imgs);
	batch_size <- imgs.shape[1];
	n_channels <- imgs.shape[2];
	img_h <- imgs.shape[3];
	img_w <- imgs.shape[4];

	# Store x for brop()
	pool[["imgs"]] <- imgs;

	# Create output array
	out_h <- (img_h - pool$win_size + 2 * pool$padding) %/% pool$stride + 1;
	out_w <- (img_w - pool$win_size + 2 * pool$padding) %/% pool$stride + 1;
	out <- array(0, c(batch_size, n_channels, out_h, out_w));

        # Perform average pooling
        imgs <- imgs / (pool$win_size)^2;
	for (b in 1:batch_size)
		for (c in 1:n_channels)
			for (y in 1:out_h)
			{
				yaux <- y * pool$stride - 1;
				pa <- max(yaux,1):min((yaux + pool$win_size - 1), img_h);
				for (x in 1:out_w)
				{
					xaux <- x * pool$stride - 1;
					pb <- max(xaux,1):min((xaux + pool$win_size - 1), img_w);
					out[b, c, y, x] <- sum(imgs[b, c, pa, pb]);
				}
			}

	list(layer = pool, y = out);
}
forward_pool <- cmpfun(forward_pool_orig);

## Backwards a Pooling Matrix (4D) to a Convolutional Matrix (4D)
##	param dy :	Array of shape (batch_size, n_channels, out_height, out_width)
##	return   :	Array of shape (batch_size, n_channels, img_height, img_width)
##	updates  :	pool_layer
backward_pool_orig <- function(pool, dy)
{
	dx <- array(0, dim(pool$imgs));
	dy <- dy / pool$win_size^2;

	dx_h <- dim(dx)[3];
	dx_w <- dim(dx)[4];

	for (i in 1:(dim(dx)[1]))
		for (c in 1:(dim(dx)[2]))
			for (y in 1:(dim(dy)[3]))
			{
				yaux <- y * pool$stride - 1;
				pa <- yaux:min((yaux + pool$win_size - 1), dx_h);
				for (x in 1:(dim(dy)[4]))
				{
					xaux <- x * pool$stride - 1;
					pb <- xaux:min((xaux + pool$win_size - 1), dx_w);
					dx[i, c, pa, pb] <- dx[i, c, pa, pb] + dy[i, c, y, x];
				}
			}
	list(layer = pool, dx = dx);
}
backward_pool <- cmpfun(backward_pool_orig);

## Updates the Pool Layer (Does nothing)
get_updates_pool <- function(pool, lr) { pool; }

## Get names of parameters and gradients (for testing functions)
pnames_pool <- function(pool) { character(0); }
gnames_pool <- function(pool) { character(0); }

## Returns a pooling layer
create_pool <- function(win_size = 3, stride = 2)
{
	list(win_size = win_size, stride = stride, padding = win_size %/% 2,
		pnames = pnames_pool, gnames = gnames_pool, forward = forward_pool,
		backward = backward_pool, get_updates = get_updates_pool);
}

## Driver for Pooling Layer
main_pool <- function()
{
	batch_size <- 1;
	n_channels <- 1;
	img_shape <- c(5, 5);
	win_size <- 3;

	x <- sample_normal(c(batch_size, n_channels, img_shape));
	layer <- create_pool(win_size = 3, stride = 2);

	ok <- check_grad(layer, x);
	if (ok)	print('Gradient check passed') else print('Gradient check failed');
}

###############################################################################
# FLATTENING LAYERS                                                           #
###############################################################################

## Creates a Flat Vector (2D) from a Convolutional Matrix (4D)
##	param x :	Array of shape (batch_size, n_channels, img_height, img_width)
##	returns :	Array of shape (batch_size, n_channels * img_height * img_width)
##	updates :	flat_layer
forward_flat <- function(flat, x)
{
	dims <- dim(x);
	flat[["shape"]] <- dims;

	batch_size <- dims[1];
	flat_dim <- prod(dims[-1]);

	y <- array(x,c(batch_size, flat_dim));
	list(layer = flat, y = y);
}

## Unflattens a Flat Vector (2D) to a Convolutional Matrix (4D)
##	param dy :	Array of shape (batch_size, n_channels * img_height * img_width)
##	return   :	Array of shape (batch_size, n_channels, img_height, img_width)
##	updates  :	flat_layer (does nothing)
backward_flat <- function(flat, dy)
{
	dx <- array(dy, flat$shape);
	list(layer = flat, dx = dx);
}

## Updates the Flat Layer (Does Nothing)
get_updates_flat <- function(flat, lr) { flat; }

## Get names of parameters and gradients (for testing functions)
pnames_flat <- function(flat) { character(0); }
gnames_flat <- function(flat) { character(0); }

## Returns a flattened layer
create_flat <- function()
{
	list(pnames = pnames_flat, gnames = gnames_flat, forward = forward_flat,
		backward = backward_flat, get_updates = get_updates_flat);
}

## Driver for Flattening Layer
main_flat <- function()
{
	batch_size <- 2;
	n_channels <- 1;
	img_shape <- c(5, 5);

	x <- sample_normal(c(batch_size, n_channels, img_shape));
	layer <- create_flat();

	ok <- check_grad(layer, x);
	if (ok)	print('Gradient check passed') else print('Gradient check failed');
}

###############################################################################
# RELU ACTIVATION LAYER                                                       #
###############################################################################

## Forwards x by setting max_0
##	param x :	Array
##	returns :	Array applied max_0
##	updates :	relu_layer
forward_relu <- function(relu, x)
{
	x[x < 0] <- 0;
	relu[["a"]] <- x;
	list(layer = relu, y = x);
}

## Returns a value activated
##	param dy :	Array
##	return   :	Array passed through (max_0)
##	updates  :	relu_layer (does nothing)
backward_relu <- function(relu, dy)
{
	dx <- dy * as.numeric(relu$a > 0);
	list(layer = relu, dx = dx);
}

## Updates the ReLU Layer (Does Nothing)
get_updates_relu <- function(relu, lr) { relu; }

## Get names of parameters and gradients (for testing functions)
pnames_relu <- function(relu) { character(0); }
gnames_relu <- function(relu) { character(0); }

## Returns a ReLU layer
create_relu <- function()
{
	list(pnames = pnames_relu, gnames = gnames_relu, forward = forward_relu,
		backward = backward_relu, get_updates = get_updates_relu);
}

###############################################################################
# LINEAR LAYER                                                                #
###############################################################################

## Forward for a linear layer
##	param x :	Numeric vector <n_visible>
##	returns :	Numeric vector <n_hidden>
##	updates :	linear_layer
forward_line <- function(line, x)
{
	line[["x"]] <- x;
	y <- (x %*% t(line$W)) %+% as.vector(line$b);

	list(layer = line, y = y);
}

## Backpropagation for a linear layer
##	param dy :	Numeric vector <n_hidden>
##	returns  :	Numeric vector <n_visible>
##	updates  :	linear_layer
backward_line <- function(line, dy)
{
	dx <- dy %*% line$W;

	line[["grad_W"]] <- t(dy) %*% line$x
	line[["grad_b"]] <- array(colSums(dy),c(1,ncol(dy)));
	
	list(layer = line, dx = dx);
}

## Updates the Linear Layer
get_updates_line <- function(line, lr)
{
        line[["W"]] = line$W - line$grad_W * lr;
        line[["b"]] = line$b - line$grad_b * lr;
	line;
}

## Get names of parameters and gradients (for testing functions)
pnames_line <- function(line) { c("W","b"); }
gnames_line <- function(line) { c("grad_W","grad_b"); }

## Returns a linear layer
create_line <- function(n_visible = 4, n_hidden = 10, scale = 0.01)
{   
	W <- scale * sample_normal(c(n_hidden, n_visible));
	b <- array(0, c(1, n_hidden));

	list(	W = W, b = b, n_visible = n_visible, n_hidden = n_hidden,
		pnames = pnames_line, gnames = gnames_line, forward = forward_line,
		backward = backward_line, get_updates = get_updates_line);
}

## Driver for Linear Layer
main_line <- function()
{
	batch_size <- 2;
	img_shape <- 100;

	x <- sample_normal(c(batch_size, img_shape));
	layer <- create_line(n_visible = 100, n_hidden = 64, scale = 0.01);

	ok <- check_grad(layer, x);
	if (ok)	print('Gradient check passed') else print('Gradient check failed');
}

###############################################################################
# SOFTMAX LAYER                                                               #
###############################################################################

## Forward through a sigmoid function
##	param x :	Numeric vector <n_visible>
##	returns :	Numeric vector <n_hidden>
##	updates :	softmax_layer
forward_soft <- function(soft, x)
{
	soft[["a"]] <- sigmoid_func(x);
	list(layer = soft, y = soft$a);
}

## Backward through the softmax layer
##	param x :	Numeric vector <n_hidden>
##	returns :	Numeric vector <n_visible>
backward_soft <- function(soft, dy)
{
	dx <- soft$a * (1 - soft$a) * dy;
	list(layer = soft, dx = dx);
}

## Updates the ReLU Layer (Does Nothing)
get_updates_soft <- function(soft, lr) { soft; }

## Get names of parameters and gradients (for testing functions)
pnames_soft <- function(soft) { character(0); }
gnames_soft <- function(soft) { character(0); }

## Returns a SoftMax layer
create_soft <- function()
{
	list(	pnames = pnames_soft, gnames = gnames_soft, forward = forward_soft,
		backward = backward_soft, get_updates = get_updates_soft);
}


###############################################################################
# CROSS-ENTROPY LOSS LAYER                                                    #
###############################################################################

## Computes the cross-entriopy for input and labels
##	param x :	Numeric vector
##	returns :	Numeric vector, Loss
forward_cell <- function(cell, x, targets)
{
	l <- -targets * log(x + 1e-08);
	l <- mean(apply(l, MARGIN = 1, sum));
	list(layer = cell, y = x, loss = l);
}

## Backpropagation of Cross-Entropy Layer
##	param x :	Numeric vector
##	returns :	Numeric vector, Loss
backward_cell <- function(cell, dy, targets)
{
	num_batches <- dim(dy)[1];
	dx <- (1.0 / num_batches) * (dy - targets);
	list(layer = cell, dx = dx);
}

## Updates the C-E Loss Layer (Does Nothing)
get_updates_cell <- function(cell, lr) { cell; }

## Get names of parameters and gradients (for testing functions)
pnames_cell <- function(cell) { character(0); }
gnames_cell <- function(cell) { character(0); }

## Returns a CrossEntropy Loss layer
create_cell <- function()
{
	list(	pnames = pnames_cell, gnames = gnames_cell, forward = forward_cell,
		backward = backward_cell, get_updates = get_updates_cell);
}

###############################################################################
# HOW TO TRAIN YOUR CNN                                                       #
###############################################################################

## Function to train the CNN
##  param training_x      : loaded dataset (rows = examples, cols = features)
##  param training_y      : loaded labels (binarized vector into rows = examples, cols = labels)
##  param training_epochs : number of epochs used for training
##  param batch_size      : size of a batch used to train the CNN
##  param learning_rate   : learning rate used for training the CNN
##  param momentum        : momentum rate used for training the CNN (Currently not used)
##  param rand_seed       : random seed for training
train_cnn <- function ( training_x, training_y, layers, training_epochs = 300,
			batch_size = 4, learning_rate = 1e-4, momentum = NULL,
			rand_seed = 1234)
{
	set.seed(rand_seed);

	num_samples <- nrow(training_x)
	num_batches <- num_samples %/% batch_size;

	loss_layer <- create_cell();

	for (epoch in 1:training_epochs)
	{
		start_time <- Sys.time();

		#confusion = ConfusionMatrix(num_classes)
		acc_loss <- NULL;		
		for (j in 1:num_batches)
		{
			# Select mini-batch
			idx <- ((j - 1) * batch_size + 1):(j * batch_size);
			batchdata <- training_x[idx,,,,drop = FALSE];		# [batch_size x n_channel x img_h x img_w]
			targets <- training_y[idx];				# [batch_size]

			# Forward
			for (i in 1:length(layers))
			{
				layer <- layers[[i]];
				forward <- layer$forward;

				aux <- forward(layer, batchdata);

				layers[[i]] <- aux$layer;
				batchdata <- aux$y;
			}
			output <- batchdata;

			# Calculate Forward Loss
			aux <- loss_layer$forward(loss_layer, output, targets);
			loss_layer <- aux$layer;
			loss <- aux$loss;

			# Calculate negdata
			aux <- loss_layer$backward(loss_layer, output, targets);
			loss_layer <- aux$layer;
			negdata <- aux$dx;			

			# Backward
			for (i in length(layers):1)
			{
				layer <- layers[[i]];
				backward <- layer$backward;

				aux <- backward(layer, negdata);

				layers[[i]] <- aux$layer;
				negdata <- aux$dx;
			}

			# Update layers
			for (i in 1:length(layers))
			{
				layer <- layers[[i]];
				get_updates <- layer$get_updates;

				layers[[i]] <- get_updates(layer, learning_rate);
			}

#			confusion.batch_add(target_batch.argmax(-1), y_probs.argmax(-1))

			acc_loss <- c(acc_loss, loss);
		}
		if (epoch %% 1 == 0)
		{
#			curr_acc = confusion.accuracy() # TODO
			print(paste("Epoch", epoch, ": Mean Loss", mean(acc_loss), sep = " "));
		}
		end_time <- Sys.time();
		print(paste('Epoch', epoch, 'took', difftime(end_time, start_time, units = "mins"), "minutes", sep=" "));
	}

	list(create_cnn(layers, loss_layer), loss = mean(acc_loss));
}

## Convolutional Neural Network (CNN). Constructor
create_cnn <- function(layers, loss_layer)
{
	list(layers = layers, loss_layer = loss_layer);
}

###############################################################################
# EXPERIMENTS: THE MNIST EXAMPLE                                              #
###############################################################################

# Load the MNIST digit recognition dataset http://yann.lecun.com/exdb/mnist/
# into R. Assume you have all 4 files and gunzip'd them creates train$n,
# train$x, train$y and test$n, test$x, test$y e.g. train$x is a 60000 x 784
# matrix, each row is one digit (28x28)
#
# Snippet authory: Brendan O'Connor https://gist.github.com/39760 - anyall.org
#
# Note: Put the MNIST data files in "./datasets/" folder

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
	train <- load_image_file('./datasets/train-images.idx3-ubyte');
	test <- load_image_file('./datasets/t10k-images.idx3-ubyte');

	train$y <- load_label_file('./datasets/train-labels.idx1-ubyte');
	test$y <- load_label_file('./datasets/t10k-labels.idx1-ubyte');

	list(train = train, test = test);
}

main <- function()
{
	# Load the MNIST dataset
	aux <- load_mnist();
	img_size <- c(28,28);

	# Set up Data as 4D matrix (batch_size, channels, H, W)
	train <- aux$train;
#	aux_x <- (train$x - mean(train$x)) / sd(train$x);
#	aux_y <- (train$y - mean(train$y)) / sd(train$y);
	training_x <- array(train$x, c(nrow(train$x), 1, img_size)) / 255;
	training_y <- binarization(train$y);

	test <- aux$test;
#	aux_x <- (test$x - mean(test$x)) / sd(test$x);
#	aux_y <- (test$y - mean(test$y)) / sd(test$y);
	testing_x <- array(test$x, c(nrow(test$x), 1, img_size)) / 255;
	testing_y <- binarization(test$y);

	# Prepare CNN
	layers <- list(
		create_conv(n_channels = 1, n_filters = 4, filter_size = 5, scale = 0.1),
		create_pool(win_size = 3, stride = 2),
		create_relu(),
		create_conv(n_channels = 4, n_filters = 16, filter_size = 5, scale = 0.1),
		create_pool(win_size = 3, stride = 2),
		create_relu(),
		create_flat(),
		create_line(n_visible = 784, n_hidden = 64, scale = 0.01),
		create_relu(),
		create_line(n_visible = 64, n_hidden = 10, scale = 0.1),
		create_soft()
	);

	# Train a CNN to learn MNIST
	cnn <- train_cnn(training_x, training_y, layers,
			 training_epochs = 10,
			 batch_size = 10,
			 learning_rate = 5e-3
	);
}

