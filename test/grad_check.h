/*---------------------------------------------------------------------------*/
/* GRADIENT CHECK for CONVOLUTIONAL NEURAL NETWORKS in C for R               */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// @date 22nd March 2017

// References:
// * Approach based on Lars Maaloee's:
//   https://github.com/davidbp/day2-Conv

#include <math.h>
#include "matrix_ops.h"

#ifndef _GRAD_CHECK
#define _GRAD_CHECK

double**** conv_op (double****, int, int, int, int, double****, int, int, int, int, int, int);
double**** forward_conv (CONV*, double****, int, int, int);
double**** backward_conv (CONV*, double****);
void get_updates_conv (CONV*, double);
void create_CONV (CONV*, int, int, int, double, int, int);
void free_CONV (CONV*);
void copy_CONV (CONV*, CONV*);

// Functions to compare gradients
int gradclose_conv (CONV* conv, double**** x , double**** y, double rtol, double atol)
{
	if (rtol == -1) rtol = 1e-05;
	if (atol == -1) atol = 1e-08;

	int is_close = 1;
	for (int b = 0; b < conv->batch_size; b++)
		for (int ci = 0; ci < conv->n_channels; ci++)
			for (int h = 0; h < conv->img_h; h++)
				for (int w = 0; w < conv->out_w; w++)
				{
					double diff = abs(x[b][ci][h][w] - y[b][ci][h][w]) - atol - rtol * (abs(x[b][ci][h][w]) + abs(y[b][ci][h][w]));
					if (diff >= 0) is_close = 0;
				}

	if (is_close == 0)
	{
		double max_re = -9e+15;
		double max_ae = -9e+15;

		for (int b = 0; b < conv->batch_size; b++)
			for (int ci = 0; ci < conv->n_channels; ci++)
				for (int h = 0; h < conv->img_h; h++)
					for (int w = 0; w < conv->out_w; w++)
					{
						double rel_error = 0;
						double denom = abs(x[b][ci][h][w]) - abs(y[b][ci][h][w]);
						if (denom != 0) rel_error = abs(x[b][ci][h][w] - y[b][ci][h][w]) / denom;

						if (rel_error > max_re) max_re = rel_error;
						if (abs(x[b][ci][h][w] - y[b][ci][h][w]) > max_ae) max_ae = abs(x[b][ci][h][w] - y[b][ci][h][w]);
					}

	        printf("rel_error=%f, abs_error=%f, rtol=%f, atol=%f", max_re, max_ae, rtol, atol);
	}
	return is_close;
}

// Check input gradient functions
double fun_cig1_conv (CONV* conv, double**** x)
{
	CONV conv_copy;
	copy_CONV(&conv_copy, conv);

	double**** y = forward_conv(&conv_copy, x, conv->n_channels, conv->img_h, conv->img_w);

	double saux = 0;
	for (int f = 0; f < conv_copy.n_filters; f++)
	{
		for (int c = 0; c < conv_copy.n_channels;c++)
		{
			for (int h = 0; h < conv_copy.out_h; h++)
			{
				for (int w = 0; w < conv_copy.out_w; w++)
					saux += y[f][c][h][w];
				free(y[f][c][h]);
			}
			free(y[f][c]);
		}
		free(y[f]);
	}
	free(y);

	free_CONV(&conv_copy);

	return saux;
}

double**** approx_fprime_conv (CONV* conv_copy, double**** x, double eps)
{
	if (eps == -1) eps = 1.4901161193847656e-08;

	// Number of elements in X
	int iters = conv_copy->batch_size * conv_copy->n_channels * conv_copy->img_h * conv_copy->img_w;

	int size_of_p_batch = sizeof(double***) * conv_copy->batch_size;
	int size_of_p_channels = sizeof(double**) * conv_copy->n_channels;
	int size_of_p_img_h = sizeof(double*) * conv_copy->img_h;
	int size_of_p_img_w = sizeof(double) * conv_copy->img_w;

	// Copy X matrix and create grad
	double**** x1 = (double****) malloc(size_of_p_batch);
	double**** x2 = (double****) malloc(size_of_p_batch);
	double**** grad = (double****) malloc(size_of_p_batch);
	for (int b = 0; b < conv_copy->batch_size; b++)
	{
		x1[b] = (double***) malloc(size_of_p_channels);
		x2[b] = (double***) malloc(size_of_p_channels);
		grad[b] = (double***) malloc(size_of_p_channels);
		for (int ci = 0; ci < conv_copy->n_channels; ci++)
		{
			x1[b][ci] = (double**) malloc(size_of_p_img_h);
			x2[b][ci] = (double**) malloc(size_of_p_img_h);
			grad[b][ci] = (double**) malloc(size_of_p_img_h);
			for (int h = 0; h < conv_copy->img_h; h++)
			{
				x1[b][ci][h] = (double*) malloc(size_of_p_img_w);
				x2[b][ci][h] = (double*) malloc(size_of_p_img_w);
				grad[b][ci][h] = (double*) malloc(size_of_p_img_w);
				for (int w = 0; w < conv_copy->out_w; w++)
				{
					x1[b][ci][h][w] = x[b][ci][h][w];
					x2[b][ci][h][w] = x[b][ci][h][w];
					grad[b][ci][h][w] = 0;
				}
			}
		}
	}

	// Compute gradient for each point in X
	for (int b = 0; b < conv_copy->batch_size; b++)
		for (int ci = 0; ci < conv_copy->n_channels; ci++)
			for (int h = 0; h < conv_copy->img_h; h++)
				for (int w = 0; w < conv_copy->out_w; w++)
				{
					double step = eps * max(abs(x[b][ci][h][w]), 1.0);
					x1[b][ci][h][w] += step;
					x2[b][ci][h][w] -= step;

					double aux1 = fun_cig1_conv (conv_copy, x1);
					double aux2 = fun_cig1_conv (conv_copy, x2);

					grad[b][ci][h][w] = (aux1 - aux2) / (2 * step);

					x1[b][ci][h][w] = x[b][ci][h][w];
					x2[b][ci][h][w] = x[b][ci][h][w];
				}

	// Free structures
	for (int b = 0; b < conv_copy->batch_size; b++)
	{
		for (int c = 0; c < conv_copy->n_channels; c++)
		{
			for (int h = 0; h < conv_copy->img_h; h++)
			{
				free(x1[b][c][h]);
				free(x2[b][c][h]);
			}
			free(x1[b][c]);
			free(x2[b][c]);
		}
		free(x1[b]);
		free(x2[b]);
	}
	free(x1);
	free(x2);

	return grad;
}

double**** fun_grad_cig1_conv (CONV* conv, double**** x)
{
	// Make a copy of layer
	CONV conv_copy;
	copy_CONV(&conv_copy, conv);

	// Go forward
	double**** y = forward_conv(&conv_copy, x, conv->n_channels, conv->img_h, conv->img_w);

	// We do nothing with Y here
	for (int f = 0; f < conv->n_filters; f++)
	{
		for (int c = 0; c < conv->n_channels; c++)
		{
			for (int h = 0; h < conv->out_h; h++)
				free(y[f][c][h]);
			free(y[f][c]);
		}
		free(y[f]);
	}
	free(y);

	// Compute gradients of param
	double**** ygrad = (double****) malloc(sizeof(double***) * conv->n_filters);
	for (int f = 0; f < conv->n_filters; f++)
	{
		ygrad[f] = (double***) malloc(sizeof(double**) * conv->n_channels);
		for (int c = 0; c < conv->n_channels; c++)
		{
			ygrad[f][c] = (double**) malloc(sizeof(double*) * conv->out_h);
			for (int h = 0; h < conv->out_h; h++)
			{
				ygrad[f][c][h] = (double*) malloc(sizeof(double) * conv->out_w);
				for (int w = 0; w < conv->out_w; w++)
					ygrad[f][c][h][w] = 1;
			}
		}
	}

	// Go backward
	double**** xgrad = backward_conv(&conv_copy, ygrad);

	// Free ygrad
	for (int f = 0; f < conv->n_filters; f++)
	{
		for (int c = 0; c < conv->n_channels; c++)
		{
			for (int h = 0; h < conv->out_h; h++)
				free(ygrad[f][c][h]);
			free(ygrad[f][c]);
		}
		free(ygrad[f]);
	}
	free(ygrad);

	// Free conv_copy
	free_CONV(&conv_copy);

	return xgrad;
}

// Return: 0 means OK, 1 means KO
int check_grad (CONV* conv, double**** x, int rand_seed, double eps, double rtol, double atol)
{
	srand(rand_seed);

	int retval = 0;

	// Part 1: Checking Output

	double**** g_approx = approx_fprime_conv(conv, x, eps);
	double**** g_true = fun_grad_cig1_conv(conv, x);

	if (gradclose_conv(conv, g_approx, g_true, rtol, atol) > 0)
	{
		printf("Incorrect Input Gradient:\n * Approx:\n");
		for (int f = 0; f < conv->n_filters; f++)
			for (int c = 0; c < conv->n_channels; c++)
				for (int h = 0; h < conv->out_h; h++)
				{
					for (int w = 0; w < conv->out_w; w++)
						printf("%f ", g_approx[f][c][h][w]);
					printf("\n");
				}
		printf(" * Approx:\n");

		for (int f = 0; f < conv->n_filters; f++)
			for (int c = 0; c < conv->n_channels; c++)
				for (int h = 0; h < conv->out_h; h++)
				{
					for (int w = 0; w < conv->out_w; w++)
						printf("%f ", g_true[f][c][h][w]);
					printf("\n");
				}
		retval = 1;
	}

	for (int f = 0; f < conv->n_filters; f++)
	{
		for (int c = 0; c < conv->n_channels; c++)
		{
			for (int h = 0; h < conv->out_h; h++)
			{
				free(g_approx[f][c][h]);
				free(g_true[f][c][h]);
			}
			free(g_approx[f][c]);
			free(g_true[f][c]);
		}
		free(g_approx[f]);
		free(g_true[f]);
	}
	free(g_approx);
	free(g_true);

	// Part 2: Checking Gradient Params
	// TODO - ...
	return retval;
}

#endif
