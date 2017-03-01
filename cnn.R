## Convolutiona Layers for DNN

## Reference: http://deeplearning.net/tutorial/lenet.html
## Reference: https://github.com/dustinstansbury/medal/blob/master/models/crbm.m
## Reference: https://github.com/davidbp/day2-Conv/blob/master/convnet_exercise_solution.ipynb

###############################################################################
# DRIVERS FOR CHECKING LAYERS                                                 #
###############################################################################

gradclose <- function(a, b, rtol = NULL, atol = NULL)
{
	rtol <- if(is.null(rtol)) 1e-05 else rtol;
	atol <- if(is.null(atol)) 1e-08 else atol;

	diff <- abs(a - b) - atol - rtol * (abs(a) + abs(b));
	is_close <- all(diff < 0);

	if (!is_close)
	{
		denom <- abs(a) - abs(b);
		mask <- denom == 0;

		rel_error <- abs(a - b) / (denom + as.numeric(mask));
		rel_error[mask] <- 0;

		rel_error <- max(rel_error);
		abs_error <- max(abs(a - b));

	        print(paste('rel_error=', rel_error, ', abs_error=', abs_error,', rtol=', rtol,', atol=', atol,sep = ""));
	}

	is_close;
}

approx_fprime <- function(x, f, eps = NULL, ...)
{
	if (is.null(eps)) eps <- 1.4901161193847656e-08;

	grad <- array(0, dim(x));
	step <- array(0, dim(x));

	for (idx in 1:length(x))
	{
		step[idx] <- eps * max(abs(x[idx]), 1.0);
		grad[idx] <- (f(x + step, ...) - f(x - step, ...)) / (2 * step[idx]);
		step[idx] <- 0;
	}
	grad;
}

check_grad <- function (layer, x0, seed = 1, eps = NULL, rtol = NULL, atol = NULL)
{
	forward <- layer$forward;
	backward <- layer$backward;
	pnames <- layer$pnames;
	gnames <- layer$gnames;

	# Check input gradient
	fun <- function(x)
	{
		y <- forward(layer, x)[["y"]];
		sum(y);
	}

	fun_grad <- function(x)
	{
		aux <- forward(layer, x);
		y <- aux$y;
		laux <- aux$layer;

		y_grad <- array(1, dim(y));
		x_grad <- backward(laux, y_grad)[["dx"]];
		x_grad;
	}

	g_approx <- approx_fprime(x0, fun, eps);
	g_true <- fun_grad(x0);

	if (!gradclose(g_approx, g_true, rtol, atol))
	{
		message(paste("Incorrect Input Gradient:", g_approx, "[<A,T>]", g_true, '\n', sep = " "));
		return(NULL);
	}

	# Check parameter gradients
	fun <- function(x, p_idx)
	{
		ns <- pnames(layer)[[p_idx]];
		param_array <- layer[[ns]];
		param_array <- param_array * 0;
		param_array <- param_array + x;
		layer[[ns]] <- param_array;
		y <- forward(layer, x0)[["y"]];
		sum(y);
	}

	fun_grad <- function(x, p_idx)
	{
		ns <- pnames(layer)[[p_idx]];
		param_array <- layer[[ns]];
		param_array <- param_array * 0;
		param_array <- param_array + x;
		layer[[ns]] <- param_array;

		aux <- forward(layer, x0);
		out <- aux$y;
		laux <- aux$layer;

		y_grad <- array(1, dim(out));
		laux <- backward(laux, y_grad)[["layer"]];

		ns <- gnames(laux)[[p_idx]];
		param_grad <- laux[[ns]];
		param_grad;
	}

	params <- pnames(layer);
	if (length(params) < 1) return();
	for (i in 1:length(params))
	{
		x <- layer[[params[i]]];
		g_true <- fun_grad(x, i);
		g_approx <- approx_fprime(x, fun, eps, i);

		if (!gradclose(g_approx, g_true, rtol, atol))
		{
			message(paste("Incorrect Parameter Gradient", i, ":", g_approx, "[<A,T>]", g_true, '\n', sep = " "));
		}
	}
}

###############################################################################
# AUXILIAR FUNCTIONS                                                          #
###############################################################################

conv2D <- function(mat, k, mode = 'valid')
{
	krow <- nrow(k);
	kcol <- ncol(k);

	mrow <- nrow(mat);
	mcol <- ncol(mat);

	out <- array(0, dim(mat));
	for(i in 1:mrow)
	{
		for(j in 1:mcol)
		{
			for(m in 1:krow)
			{
				mm <- krow - m + 1;
				for(n in 1:kcol)
				{
					nn <- kcol - n + 1;

					ii <- i + (m - (krow %/% 2)) - 1;
					jj <- j + (n - (kcol %/% 2)) - 1;

					if( ii > 0 && ii <= mrow && jj > 0 && jj <= mcol)
						out[i,j] <- out[i,j] + mat[ii,jj] * k[mm,nn];
				}
			}
		}
	}

	if (mode == 'valid')
	{
		cut_y <- nrow(k) %/% 2;
		cut_x <- ncol(k) %/% 2;
		out <- out[(cut_y + 1):(nrow(out) - cut_y), (cut_x + 1):(ncol(out) - cut_x)];
	}

	out;
}

img_padding <- function(img, pad_x, pad_y)
{
	dims <- dim(img);
	imgs_pad <- array(0, c(dims[1] + 2 * pad_x, dims[2] + 2 * pad_y));

	aux <- cbind(img, array(0, c(nrow(img), pad_x)));
	aux <- cbind(array(0, c(nrow(aux), pad_x)), aux);
	aux <- rbind(aux, array(0, c(pad_y, ncol(aux))));
	aux <- rbind(array(0, c(pad_y, ncol(aux))), aux);
	aux;
}

###############################################################################
# CONVOLUTIONAL LAYERS                                                        #
###############################################################################

## Performs the convolution
##	param imgs: <batch_size, img_n_channels, img_height, img_width>
##	param filters: <n_filters, n_channels, win_height, win_width>
##	param padding: <padding_y, padding_x>
conv_bc01 <- function(imgs, filters, padding)
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

	# Pad input images
	imgs_pad <- array(0, dim(imgs) + c(0, 0, 2*(pad_x), 2*(pad_y)));
	for (i in 1:dim(imgs)[1])
		for (j in 1:dim(imgs)[2])
			imgs_pad[i,j,,] <- img_padding(imgs[i,j,,], pad_x, pad_y);

	# Perform convolution #FIXME - Optimize with apply
	for (b in 1:batch_size)
		for (f in 1:n_filters)
			for (c in 1:n_channels)
			{
				cnv <- conv2D(imgs_pad[b,c,,], filters[f,c,,]);
				out_1[b,f,,] <- out_1[b,f,,] + cnv;
			}
	return(out_1);
}

## Performs Forward Propagation
##	param x :	Array of shape (batch_size, n_channels, img_height, img_width)
##	return  :	Array of shape (batch_size, n_filters, out_height, out_width)
##	updates :	conv_layer
forward_conv <- function(conv, x)
{      
	# Save "x" for back-propagation
	conv[["x"]] <- x;

        # Performs convolution
	y <- conv_bc01(x, conv$W, conv$padding);

	for (b in 1:dim(y)[1]) y[b,,,] <- y[b,,,] + conv$b[1,,1,1];
	list(layer = conv, y = y);
}

## Performs Backward Propagation
##	param dy :	4D-image
##	return   :	image
##	updates  :	conv_layer
backward_conv <- function(conv, dy)
{
        # Flip weights
	w <- conv$W[,, (conv$w_shape[3]:1), (conv$w_shape[4]:1), drop = FALSE];

        # Transpose channel/filter dimensions of weights
	w <- aperm(w, c(2, 1, 3, 4));

        # Propagate gradients to x
        dx <- conv_bc01(dy, w, conv$padding)

        # Propagate gradients to weights and gradients to bias
	x_pad <- array(0, dim(conv$x) + c(0, 0, 2 * conv$padding));
	for (i in 1:dim(x_pad)[1])
		for (j in 1:dim(x_pad)[2])
			x_pad[i,j,,] <- img_padding(conv$x[i,j,,], conv$padding[1], conv$padding[2]);

	grad_W <- array(0, dim(conv$W));
	for (b in 1:dim(dy)[1])
		for (f in 1:dim(conv$W)[1])
			for (c in 1:dim(conv$W)[2])
				grad_W[f,c,,] <- grad_W[f,c,,] + conv2D(x_pad[b,c,,], dy[b,f,,]);

	conv[["grad_W"]] <- grad_W[,, (conv$w_shape[3]:1), (conv$w_shape[4]:1), drop = FALSE];
	conv[["grad_b"]] <- array(apply(dy, MARGIN = 2, sum), dim(conv$b));

	list(layer = conv, dx = dx);
}

## Updates the Convolutional Layer
get_updates_conv <- function(conv, lr)
{
	conv[["W"]] = conv$W - conv$grad_W * lr;
	conv[["b"]] = conv$b - conv$grad_b * lr;
	conv;
}

## Get names of parameters and gradients (for testing functions)
pnames_conv <- function(conv) { c("W","b"); }
gnames_conv <- function(conv) { c("grad_W","grad_b"); }


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
	batch_size <- 2;
	n_channels <- 1;
	img_shape <- c(5, 5);
	n_filters <- 2;
	filter_size <- 3;

	border_mode <- 'same';

	x <- sample_normal(c(batch_size, n_channels, img_shape));
	layer <- create_conv(n_channels, n_filters, filter_size, border_mode = border_mode);

	check_grad(layer, x);
	print('Gradient check passed')
}

###############################################################################
# POOLING LAYERS                                                              #
###############################################################################

## Forwards a Pooling Matrix (4D) from a Convolutional Matrix (4D)
##	param x :	Array of shape (batch_size, n_channels, img_height, img_width)
##	returns :	Array of shape (batch_size, n_channels, out_height, out_width)
##	updates :	pool_layer
forward_pool <- function(pool, imgs)
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
				for (x in 1:out_w)
				{
					xaux <- x * pool$stride - 1;
					pa <- max(yaux,1):min((yaux + pool$win_size - 1), img_h);
					pb <- max(xaux,1):min((xaux + pool$win_size - 1), img_w);
					win <- imgs[b, c, pa, pb];
					out[b, c, y, x] <- sum(win);
				}
			}

	list(layer = pool, y = out);
}

## Backwards a Pooling Matrix (4D) to a Convolutional Matrix (4D)
##	param dy :	Array of shape (batch_size, n_channels, out_height, out_width)
##	return   :	Array of shape (batch_size, n_channels, img_height, img_width)
##	updates  :	pool_layer
backward_pool <- function(pool, dy)
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
				for (x in 1:(dim(dy)[4]))
				{
					xaux <- x * pool$stride - 1;
					pa <- yaux:min((yaux + pool$win_size - 1), dx_h);
					pb <- xaux:min((xaux + pool$win_size - 1), dx_w);
					dx[i, c, pa, pb] <- dx[i, c, pa, pb] + dy[i, c, y, x];
				}
			}
	list(layer = pool, dx = dx);
}

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

	check_grad(layer, x);
	print('Gradient check passed');
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
	flat[["shape"]] <- dim(x);
	y <- array(x,prod(dim(x)))
	list(layer = flat, y = y);
}

## Unflattens a Flat Vector (2D) to a Convolutional Matrix (4D)
##	param dy :	Array of shape (batch_size, n_channels * img_height * img_width)
##	return   :	Array of shape (batch_size, n_channels, img_height, img_width)
##	updates  :	flat_layer (does nothing)
backward_flat <- function(flat, delta_in)
{
	dx <- array(delta_in, flat$shape);
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

	check_grad(layer, x);
	print('Gradient check passed');
}


###############################################################################
# CNN FUNCTIONS                                                               #
###############################################################################

source("dnn.R");

## Convolutional Neural Network (CNN). Constructor
create_cnn <- function(n_visible = 1, rand_seed = 1234, ...)
{
	set.seed(rand_seed);

	# Posterior MLP
	dnn <- create_dnn(n_visible = 1, ...);

	list(dnn = dnn);
}

## Network description:
##
##	(Conv -> Pool -> Relu)+ -> Flat -> MLP

