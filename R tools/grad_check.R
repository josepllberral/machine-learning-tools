###############################################################################
# GRADIENT CHECK for CONVOLUTIONAL NEURAL NETWORKS in R                       #
###############################################################################

## @author Josep Ll. Berral (Barcelona Supercomputing Center)

## @date 3rd March 2017

## References:
## * Approach based on Lars Maaloee's:
##   https://github.com/davidbp/day2-Conv

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

check_grad <- function (layer, x0, seed = 1, eps = NULL, rtol = NULL, atol = NULL, ...)
{
	forward <- layer$forward;
	backward <- layer$backward;
	pnames <- layer$pnames;
	gnames <- layer$gnames;

	# Check input gradient
	fun <- function(x, ...)
	{
		y <- forward(layer, x, ...)[["y"]];
		sum(y);
	}

	fun_grad <- function(x, ...)
	{
		aux <- forward(layer, x, ...);
		y <- aux$y;
		laux <- aux$layer;

		y_grad <- array(1, dim(y));
		x_grad <- backward(laux, y_grad, ...)[["dx"]];

		x_grad;
	}

	g_approx <- approx_fprime(x0, fun, eps, ...);
	g_true <- fun_grad(x0, ...);

	is_close <- gradclose(g_approx, g_true, rtol, atol);

	if (!is_close)
	{
#		message(paste("Incorrect Input Gradient:", g_approx, "[<A,T>]", g_true, '\n', sep = " "));
		message("Incorrect Input Gradient\n* Approx:\n");
		print(g_approx[1,1,,]);	
		message("True:\n");
		print(g_true[1,1,,]);
		return(FALSE);
	}

	# Check parameter gradients
	fun <- function(x, p_idx, ...)
	{
		ns <- (pnames())[[p_idx]];
		param_array <- layer[[ns]];
		param_array <- param_array * 0;
		param_array <- param_array + x;
		layer[[ns]] <- param_array;
		y <- forward(layer, x0, ...)[["y"]];
		sum(y);
	}

	fun_grad <- function(x, p_idx, ...)
	{
		ns <- (pnames())[[p_idx]];
		param_array <- layer[[ns]];
		param_array <- param_array * 0;
		param_array <- param_array + x;
		layer[[ns]] <- param_array;

		aux <- forward(layer, x0, ...);
		out <- aux$y;
		laux <- aux$layer;

		y_grad <- array(1, dim(out));
		laux <- backward(laux, y_grad, ...)[["layer"]];

		gnames <- laux$gnames;
		ns <- (gnames())[[p_idx]];
		param_grad <- laux[[ns]];
		param_grad;
	}

	params <- layer$pnames();
	if (length(params) < 1) return();
	for (i in 1:length(params))
	{
		x <- layer[[params[i]]];
		g_true <- fun_grad(x, i, ...);
		g_approx <- approx_fprime(x, fun, eps, i, ...);

		if (!gradclose(g_approx, g_true, rtol, atol))
		{
			message(paste("Incorrect Parameter Gradient", i, ":", g_approx, "[<A,T>]", g_true, '\n', sep = " "));
			return(FALSE);
		}
	}
	return(TRUE);
}

