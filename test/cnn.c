/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// @date 22nd March 2017

// References:
// * Approach based on Lars Maaloee's:
//   https://github.com/davidbp/day2-Conv
// * Also from LeNet (deeplearning.net)
//   http://deeplearning.net/tutorial/lenet.html

// Mocap data:
// The MNIST digit recognition dataset http://yann.lecun.com/exdb/mnist/

// Compile using "R CMD SHLIB crbm.c"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>

#include "matrix_ops.h"

typedef struct {
	int batch_size;
	int filter_size;
	int n_filters;
	int n_channels;
	int win_h;
	int win_w;
	int mat_h;
	int mat_w;
	double**** W;
	double* b;
	double**** grad_W;
	double* grad_b;
	int pad_y;
	int pad_x;
	double**** img;
	int img_h;
	int img_w;
	int n_channels_img;
	int out_h;
	int out_w;
} CONV;

/*---------------------------------------------------------------------------*/
/* AUXILIAR FUNCTIONS                                                        */
/*---------------------------------------------------------------------------*/

inline int max(int a, int b) { return a > b ? a : b; }
inline int min(int a, int b) { return a < b ? a : b; }

// Function to perform the convolution
//  param mode : 1 = "valid"
double** conv2D(double** mat, int mrow, int mcol, double** k, int krow, int kcol, int mode)
{
	int krow_h = krow / 2;
	int kcol_h = kcol / 2;

	double** out = (double**) malloc(sizeof(double*) * mrow);
	for(int i = 0; i < mrow; ++i)
	{
		out[i] = (double*) malloc(sizeof(double) * mcol);
		for(int j = 0; j < mcol; ++j)
		{
			double acc = 0;
			for(int m = 0; m < krow; ++m)
			{
				int mm = krow - 1 - m;
				int ii = i + (m - krow_h);

				if (ii >= 0 || ii < mrow)
					for(int n = 0; n < kcol; ++n)
					{
						int nn = kcol - 1 - n;
						int jj = j + (n - kcol_h);

						if (jj >= 0 && jj < mcol) acc += mat[ii][jj] * k[mm][nn];
					}
			}
			out[i][j] = acc;
		}
	}

	if (mode == 1)
	{
		int cut_y = krow_h;
		int cut_x = kcol_h;

		int len_y = max(krow,mrow) - min(krow,mrow);
		int len_x = max(kcol,mcol) - min(kcol,mcol);

		double** tmp = (double**) malloc(sizeof(double*) * len_y);
		for (int y = 0; y < len_y; y++)
		{
			tmp[y] = (double*) malloc(sizeof(double) * len_x);
			for (int x = 0; x < len_x; x++)
				tmp[y][x] = out[cut_y + y][cut_x + x];
		}

		matrix_free(out, mrow);
		out = tmp;
	}

	return out;
}

// Function to pad images
double** img_padding (double** img, int irow, int icol, int pad_y, int pad_x)
{
	double** out = (double**) malloc(sizeof(double*) * (irow + 2 * pad_y));
	for (int i = 0; i < (irow + 2 * pad_y); i++)
		out[i] = (double*) calloc((icol + 2 * pad_x), sizeof(double));

	for(int i = pad_y; i < irow - pad_y; i++)
		for(int j = pad_x; j < icol - pad_x; j++)
			out[i][j] = img[i - pad_y][j - pad_x];

	return out;
}

// Function to binarize a vector
//  param vec    : vector of integers (factors and strings must be numerized before)
//  param vlen   : length of vector
//  param vclass : number of classes in vec => lenght(unique(vec))
int** binarization_cnn(int* vec, int vlen, int vclass)
{
	int** out = (int**) malloc(sizeof(int*) * vlen);
	for (int i = 0; i < vlen; i++)
	{
		out[i] = (int*) calloc(vclass, sizeof(int));
		out[i][vec[i]] = 1;
	}

	return out;
}

/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL LAYERS                                                      */
/*---------------------------------------------------------------------------*/

// This function performs the convolution
//   param imgs     : <batch_size, img_n_channels, img_height, img_width>
//   param filters  : <n_filters, n_channels, win_height, win_width>
//   param padding  : <padding_y, padding_x>
//   param out_dims : <out_h, out_w>
double**** conv_op (double**** imgs, int batch_size, int n_channels_img,
	int img_h, int img_w, double**** filters, int n_filters, int n_channels,
	int win_h, int win_w, int pad_y, int pad_x, int out_h, int out_w)
{
	if (!(n_channels == n_channels_img))
	{
		printf("ERROR: Mismatch in Number of Channels\n");
		return NULL;
	}

	// Create output array and Prepare padded image for convolution
	int size_of_p_out_h = sizeof(double*) * out_h;
	int size_of_p_n_filters = sizeof(double**) * n_filters;
	int size_of_p_n_chan_img = sizeof(double**) * n_channels_img;

	double**** out = (double****) malloc(sizeof(double***) * batch_size);
	double**** imgs_pad = (double****) malloc(sizeof(double***) * batch_size);
	for (int i = 0; i < batch_size; i++)
	{
		out[i] = (double***) malloc(size_of_p_n_filters);
		for (int j = 0; j < n_filters; j++)
		{
			out[i][j] = (double**) malloc(size_of_p_out_h);
			for (int k = 0; k < out_h; k++)
				out[i][j][k] = (double*) calloc(out_w, sizeof(double));
		}

		imgs_pad[i] = (double***) malloc(size_of_p_n_chan_img);
		for (int j = 0; j < n_channels_img; j++)
			imgs_pad[i][j] = img_padding(imgs[i][j], img_h, img_w, pad_y, pad_x);
	}

	// Perform convolution
	int pad_h = img_h + 2 * pad_y;
	int pad_w = img_w + 2 * pad_x;
	for (int b = 0; b < batch_size; b++)
		for (int f = 0; f < n_filters; f++)
			for (int c = 0; c < n_channels; c++)
			{
				double** conv_aux = conv2D(imgs_pad[b][c], pad_h, pad_w, filters[f][c], win_h, win_w, 1);
				for (int i = 0; i < out_h; i++)
					for (int j = 0; j < out_w; j++)
						out[b][f][i][j] += conv_aux[i][j];
			}

	// Free auxiliar structures
	for (int i = 0; i < batch_size; i++)
	{
		for (int j = 0; j < n_filters; j++)
			matrix_free(imgs_pad[i][j], img_h + 2 * pad_y);
		free(imgs_pad[i]);
	}
	free(imgs_pad);

	return out;
}

// This function performs Forward Propagation
//    param x : Array of shape (batch_size, n_channels_img, img_height, img_width)
//    return  : Array of shape (batch_size, n_filters, out_height, out_width)
//    updates : conv_layer
double**** forward_conv (CONV* conv, double**** x, int n_channels_img,
	int img_height, int img_width)
{
	// Save "x" for back-propagation
	if (conv->img != NULL)
	{
		for (int b = 0; b < conv->batch_size; b++)
		{
			for (int ci = 0; ci < conv->n_channels_img; ci++)
				matrix_free(conv->img[b][ci], conv->img_h);
			free(conv->img[b]);
		}
		free(conv->img);
	}
	conv->img = x;
	conv->img_h = img_height;
	conv->img_w = img_width;
	conv->n_channels_img = n_channels_img;

	conv->out_h = (conv->img_h - conv->win_h + 2 * conv->pad_y) + 1;
	conv->out_w = (conv->img_w - conv->win_w + 2 * conv->pad_x) + 1;

	// Performs convolution
	double**** y = conv_op(x, conv->batch_size, conv->n_channels_img,
		conv->img_h, conv->img_w, conv->W, conv->n_filters,
		conv->n_channels, conv->win_h, conv->win_w, conv->pad_y,
		conv->pad_x, conv->out_h, conv->out_w);

	for (int b = 0; b < conv->batch_size; b++)
		for (int f = 0; f < conv->n_filters; f++)
			for (int h = 0; h < conv->out_h; h++)
				for (int w = 0; w < conv->out_w; w++)
					y[b][f][h][w] += conv->b[f];
	return y;
}

// This function performs Backward Propagation
//    param dy : Array of shape (batch_size, n_filters, out_height, out_width)
//    return   : Array of shape (batch_size, n_channels_img, img_height, img_width)
//    updates  : conv_layer
double**** backward_conv (CONV* conv, double**** dy)
{
	int size_of_p_n_filters = sizeof(double***) * conv->n_filters;
	int size_of_p_n_chan_img = sizeof(double**) * conv->n_channels_img;
	int size_of_p_n_channels = sizeof(double**) * conv->n_channels;
	int size_of_p_batch_size = sizeof(double**) * conv->batch_size;
	int size_of_p_mat_h = sizeof(double*) * conv->mat_h;
	int size_of_p_mat_w = sizeof(double) * conv->mat_w;

        // Flip weights & Transpose channel/filter dimensions of weights
	double**** waux = (double****) malloc(size_of_p_n_filters);
	for (int f = 0; f < conv->n_filters; f++)
	{
		waux[f] = (double***) malloc(size_of_p_batch_size);
		for (int b = 0; b < conv->batch_size; b++)
		{
			waux[f][b] = (double**) malloc(size_of_p_mat_h);
			for (int h = 0; h < conv->mat_h; h++)
			{
				waux[f][b][h] = (double*) malloc(size_of_p_mat_w);
				for (int w = 0; w < conv->mat_w; w++)
				{
					int hflip = conv->mat_h - h - 1;
					int wflip = conv->mat_w - w - 1;
					waux[f][b][h][w] = conv->W[b][f][hflip][wflip];
				}
			}
		}
	}

	double**** dx = conv_op(dy, conv->batch_size, conv->n_channels_img,
		conv->img_h, conv->img_w, waux, conv->n_filters,
		conv->n_channels, conv->win_h, conv->win_w, conv->pad_y,
		conv->pad_x, conv->out_h, conv->out_w);

	// Prepares padded image for convolution
	double**** x_pad = (double****) malloc(sizeof(double***) * conv->batch_size);
	for (int b = 0; b < conv->batch_size; b++)
	{
		x_pad[b] = (double***) malloc(size_of_p_n_chan_img);
		for (int ci = 0; ci < conv->n_channels_img; ci++)
			x_pad[b][ci] = img_padding(conv->img[b][ci], conv->img_h, conv->img_w, conv->pad_y, conv->pad_x);
	}

	// Propagate gradients to weights and gradients to bias
	double**** grad_W = (double****) malloc(size_of_p_n_filters);
	double* grad_b = (double*) calloc(conv->n_filters, sizeof(double));

	for (int f = 0; f < conv->n_filters; f++)
	{
		grad_W[f] = (double***) malloc(size_of_p_n_channels);
		for (int c = 0; c < conv->n_channels; c++)
		{
			grad_W[f][c] = (double**) malloc(size_of_p_mat_h);
			for (int h = 0; h < conv->mat_h; h++)
				grad_W[f][c][h] = (double*) calloc(conv->mat_w, sizeof(double));
		}
	}

	int pad_h = conv->img_h + 2 * conv->pad_y;
	int pad_w = conv->img_w + 2 * conv->pad_x;
	for (int b = 0; b < conv->batch_size; b++)
		for (int f = 0; f < conv->n_filters; f++)
		{
			for (int c = 0; c < conv->n_channels; c++)
			{
				double** conv_aux = conv2D(x_pad[b][c], pad_h, pad_w, dy[b][f], conv->out_h, conv->out_w, 1);
				for (int h = 0; h < conv->mat_h; h++)
					for (int w = 0; w < conv->mat_w; w++)
						grad_W[f][c][h][w] += conv_aux[h][w];
				matrix_free(conv_aux, conv->mat_h);
			}
			for (int h = 0; h < conv->out_h; h++)
				for (int w = 0; w < conv->out_w; w++)
					grad_b[f] += dy[b][f][h][w];
		}

	// Flip back grad_W
	for (int f = 0; f < conv->n_filters; f++)
		for (int c = 0; c < conv->n_channels; c++)
			for (int h = 0; h < conv->mat_h; h++)
				for (int w = 0; w < conv->mat_w; w++)
				{
					int hflip = conv->mat_h - h - 1;
					int wflip = conv->mat_w - w - 1;
					double aux = grad_W[f][c][h][w];
					grad_W[f][c][h][w] = grad_W[f][c][hflip][wflip];
					grad_W[f][c][hflip][wflip] = aux;
				}

	// Update gradient values
	for (int f = 0; f < conv->n_filters; f++)
	{
		for (int c = 0; c < conv->n_channels; c++)
			matrix_free(conv->grad_W[f][c], conv->mat_h);
		free(conv->grad_W[f]);
	}
	free(conv->grad_W);
	conv->grad_W = grad_W;

	free(conv->grad_b);
	conv->grad_b = grad_b;

	// Free auxiliar structures
	for (int i = 0; i < conv->batch_size; i++)
	{
		for (int j = 0; j < conv->n_channels_img; j++)
			matrix_free(x_pad[i][j], conv->img_h + 2 * conv->pad_y);
		free(x_pad[i]);
	}
	free(x_pad);

	return dx;
}

// Updates the Convolutional Layer
void get_updates_conv (CONV* conv, double lr)
{
	for (int f = 0; f < conv->n_filters; f++)
	{
		for (int c = 0; c < conv->n_channels; c++)
			for (int h = 0; h < conv->mat_h; h++)
				for (int w = 0; w < conv->mat_w; w++)
					conv->W[f][c][h][w] -= conv->grad_W[f][c][h][w] * lr;
		conv->b[f] -= conv->grad_b[f] * lr;
	}
}

// Initializes a convolutional layer
void create_CONV (CONV* conv, int n_channels, int n_filters, int filter_size,
	double scale, int border_mode, int batch_size) // scale = 0.01, border_mode = 1 "valid"
{
	conv->batch_size = batch_size;
	conv->filter_size = filter_size;

	conv->n_filters = n_filters;
	conv->n_channels = n_channels;
	conv->win_h = filter_size;
	conv->win_w = filter_size;

	conv->mat_h = filter_size;
	conv->mat_w = filter_size;

	int size_of_p_n_filters = sizeof(double**) * n_filters;

	conv->W = (double****) malloc(sizeof(double***) * n_filters);
	for (int f = 0; f < n_filters; f++)
	{
		conv->W[f] = (double***) malloc(size_of_p_n_filters);
		for (int c = 0; c < n_channels; c++)
			conv->W[f][c] = matrix_normal(filter_size, filter_size, 0, 1, scale);
	}
	conv->b = (double*) calloc(n_filters, sizeof(double));

	int padding = 0;
	if (border_mode == 1) padding <- 0; 			// 'valid'
	if (border_mode == 2) padding <- filter_size / 2;	// 'same'
	if (border_mode == 3) padding <- filter_size - 1;	// 'full'

	conv->pad_y = padding;
	conv->pad_x = padding;
}

// Destructor of Convolutional Layer
void free_CONV (CONV* conv)
{
	// Free last batch of images
	if (conv->img != NULL)
	{
		for (int b = 0; b < conv->batch_size; b++)
		{
			for (int ci = 0; ci < conv->n_channels_img; ci++)
				matrix_free(conv->img[b][ci], conv->img_h);
			free(conv->img[b]);
		}
		free(conv->img);
	}

	// Free weights matrix and bias
	for (int f = 0; f < conv->n_filters; f++)
	{
		for (int ci = 0; ci < conv->n_channels; ci++)
			matrix_free(conv->W[f][ci], conv->mat_h);
		free(conv->W[f]);
	}
	free(conv->W);
	free(conv->b);
}

