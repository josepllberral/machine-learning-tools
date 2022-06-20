/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C                                               */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations

// Reference for FFT Convolution:
// * URL: https://github.com/jeremyfix/FFTConvolution

// Reference for Factorization code:
// * Code adapted from gsl/fft/factorize.c

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* AUXILIAR FUNCTIONS FOR FFT CONVOLUTION                                    */
/*---------------------------------------------------------------------------*/

void factorize (const int n, int* n_factors, int factors[], int* implemented_factors)
{
	int i = 0;
	int nf = 0;
	int ntest = n;
	int factor;

	int product = 1;

	if (n == 0)
	{
		printf("Length n must be positive integer\n");
		return;
	}
	if (n == 1)
	{
		factors[0] = 1;
		*n_factors = 1;
		return;
	}

	while (implemented_factors[i] && ntest != 1)
	{
		factor = implemented_factors[i];
		while ((ntest % factor) == 0)
		{
			ntest = ntest / factor;
			factors[nf] = factor;
			nf++;
		}
		i++;
	}

	if(ntest != 1)
	{
		factors[nf] = ntest;
		nf++;
	}

	for (i = 0; i < nf; i++) product *= factors[i];
	if (product != n) printf("Factorization failed!\n");

	*n_factors = nf;
}

int is_optimal(int n, int* implemented_factors)
{
	int i = 0;
	int nf = 0;
	int factors[64];

	// Check that N is not multiple of 4*4*4*2
	if (n % 4*4*4*2 == 0) return 0;

	// Check if last factor belongs to GSL_FACTORS
	factorize(n, &nf, factors, implemented_factors);
	while (implemented_factors[i])
	{
		if (factors[nf - 1] == implemented_factors[i]) return 1;
		i++;
	}
	return 0;
}

int find_closest_factor(int n, int* implemented_factor)
{
	int j = n + 1;

	if (is_optimal(n, implemented_factor) == 1) return n;
	else
	{
		while(is_optimal(j, implemented_factor) == 0) j++;
		return j;
	}
}

// Function to perform the convolution, using FFT
//  param mode : 1 = "valid"
gsl_matrix* conv2D_fft (gsl_matrix* mat, gsl_matrix* k, int mode)
{
	// Get matrices sizes
	int mrow = mat->size1;
	int mcol = mat->size2;
	int krow = k->size1;
	int kcol = k->size2;

	gsl_fft_complex_workspace *ws_column, *ws_line;
	gsl_fft_complex_wavetable *wv_column, *wv_line;

	gsl_matrix* fft;
	gsl_matrix* fft_copy;
	int frow, fcol;

	gsl_matrix* out;
	int drow, dcol;

	// Auxiliar vars
	double re_h, im_h, re_hs, im_hs;
	int ii, jj;
	int GSL_FACTORS[7] = {7,6,5,4,3,2,0};

	if (mode == 1)
	{
		if(krow > mrow || kcol > mcol) return NULL;
		else
		{
			frow = find_closest_factor(mrow, GSL_FACTORS);
			fcol = find_closest_factor(mcol, GSL_FACTORS);
			drow = mrow - krow + 1;
			dcol = mcol - kcol + 1;
		}
	}
	else
	{
		printf("Warning: Mode %d not implemented yet. Performing classic Convolution.", mode);
		return conv2D(mat, k, mode);
	}

	// FFT Matrices are x2 (real, imaginary)
	fft       = gsl_matrix_alloc(frow, 2 * fcol);
	fft_copy  = gsl_matrix_alloc(frow, 2 * fcol);
	ws_column = gsl_fft_complex_workspace_alloc(frow);
	ws_line   = gsl_fft_complex_workspace_alloc(fcol);
	wv_column = gsl_fft_complex_wavetable_alloc(frow);
	wv_line   = gsl_fft_complex_wavetable_alloc(fcol);
	out       = gsl_matrix_calloc(drow, dcol);

	gsl_matrix_set_zero(fft);

	// Perform FFT Convolution (Input = Real part, Kernel = Imaginary part)
	for(int i = 0 ; i < mrow ; i++)
	{
		ii = i % frow;
		for(int j = 0 ; j < mcol ; j++)
		{
			jj = (j % fcol) * 2;
			gsl_matrix_set(fft, ii, jj, gsl_matrix_get(fft, ii, jj) + gsl_matrix_get(mat, i, j));
		}
	}
	for(int i = 0 ; i < krow ; i++)
	{
		ii = i % frow;
		for(int j = 0 ; j < kcol ; j++)
		{
			jj = (j % fcol) * 2 + 1;
			gsl_matrix_set(fft, ii, jj, gsl_matrix_get(fft, ii, jj) + gsl_matrix_get(k, i, j));
		}
	}

	// FFT Forward
	for(int i = 0 ; i < frow ; i++)
		gsl_fft_complex_forward(&fft->data[i * 2 * fcol], 1, fcol, wv_line, ws_line);
	for(int j = 0 ; j < fcol ; j++)
		gsl_fft_complex_forward(&fft->data[2 * j], fcol, frow, wv_column, ws_column);

	// Element-wise product
	for(int i = 0 ; i < frow ; i++)
	{
		ii = (frow - i) % frow;
		for(int j = 0 ; j < fcol ; j++)
		{
			jj    = (2 * (fcol - j)) % (2 * fcol);
			re_h  = gsl_matrix_get(fft, i, 2 * j);
			im_h  = gsl_matrix_get(fft, i, 2 * j + 1);
			re_hs = gsl_matrix_get(fft, ii, jj);
			im_hs = -1 * gsl_matrix_get(fft, ii, jj + 1);

			gsl_matrix_set(fft_copy, i, 2 * j, 0.5 * ((re_h * im_h) - (re_hs * im_hs)));
			gsl_matrix_set(fft_copy, i, 2 * j + 1, -0.25 * ((re_h * re_h) - (im_h * im_h) - (re_hs * re_hs) + (im_hs * im_hs)));
		}
	}

	// Inverse FFT
	for(int i = 0 ; i < frow ; i++)
		gsl_fft_complex_inverse(&fft_copy->data[i * 2 * fcol], 1, fcol, wv_line, ws_line);
	for(int j = 0 ; j < fcol ; j++)
		gsl_fft_complex_inverse(&fft_copy->data[2 * j], fcol, frow, wv_column, ws_column);

	// Prepare Results Matrix ([drow x dcol], starting by [krow-1, kcol-1])
	for(int i = 0 ; i < drow ; i++) for(int j = 0 ; j < dcol ; j++)
		gsl_matrix_set(out, i, j, gsl_matrix_get(fft_copy, (i + krow - 1), 2 * (j + kcol - 1)));

	// Free Temporary Elements
	gsl_fft_complex_workspace_free(ws_column);
	gsl_fft_complex_workspace_free(ws_line);
	gsl_fft_complex_wavetable_free(wv_column);
	gsl_fft_complex_wavetable_free(wv_line);
	gsl_matrix_free(fft);
	gsl_matrix_free(fft_copy);

	return out;
}

/*---------------------------------------------------------------------------*/
/* AUXILIAR FUNCTIONS FOR TRUE CONVOLUTION                                   */
/*---------------------------------------------------------------------------*/

// Function to perform the convolution
//  param mode : 1 = "valid"
gsl_matrix* conv2D (gsl_matrix* mat, gsl_matrix* k, int mode)
{
	// Get matrices sizes
	int mrow = mat->size1;
	int mcol = mat->size2;
	int krow = k->size1;
	int kcol = k->size2;

	int krow_h = krow / 2;
	int kcol_h = kcol / 2;

	gsl_matrix* out = gsl_matrix_calloc(mrow, mcol);
	for(int i = 0; i < mrow; ++i)
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

						if (jj >= 0 && jj < mcol) acc += gsl_matrix_get (mat, ii, jj) * gsl_matrix_get (k, mm, nn);
					}
			}
			gsl_matrix_set(out, i, j, acc);
		}

	if (mode == 1)
	{
		int cut_y = krow_h;
		int cut_x = kcol_h;

		int len_y = max(krow,mrow) - min(krow,mrow) + 1;
		int len_x = max(kcol,mcol) - min(kcol,mcol) + 1;

		gsl_matrix* tmp = gsl_matrix_calloc(len_y, len_x);
		for (int y = 0; y < len_y; y++)
			for (int x = 0; x < len_x; x++)
				gsl_matrix_set(tmp, y, x, gsl_matrix_get(out, cut_y + y, cut_x + x));

		gsl_matrix_free(out);
		out = tmp;
	}

	return out;
}

// Function to pad images
gsl_matrix* img_padding (gsl_matrix* img, int pad_y, int pad_x)
{
	int irow = img->size1;
	int icol = img->size2;

	gsl_matrix* out_p = gsl_matrix_calloc(irow + 2 * pad_y, icol);
	for(int i = 0; i < irow; i++)
	{
		gsl_vector* aux = gsl_vector_alloc(icol);
		gsl_matrix_get_row(aux, img, i);
		gsl_matrix_set_row(out_p, i + pad_y, aux);
		gsl_vector_free(aux);
	}

	gsl_matrix* out = gsl_matrix_calloc(irow + 2 * pad_y, icol + 2 * pad_x);
	for(int j = 0; j < icol; j++)
	{
		gsl_vector* aux = gsl_vector_alloc(irow + 2 * pad_y);
		gsl_matrix_get_col(aux, out_p, j);
		gsl_matrix_set_col(out, j + pad_x, aux);
		gsl_vector_free(aux);
	}

	gsl_matrix_free(out_p);

	return out;
}

/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL LAYERS                                                      */
/*---------------------------------------------------------------------------*/

// This function performs the convolution
//   param imgs     : <batch_size, img_n_channels, img_height, img_width>
//   param filters  : <n_filters, n_channels, filter_size, filter_size>
//   param padding  : <padding_y, padding_x>
gsl_matrix*** conv_op (	gsl_matrix*** imgs, int batch_size, int n_channels_img,
			gsl_matrix*** filters, int n_filters, int n_channels,
			int pad_y, int pad_x)
{
	// Get image shapes
	int img_h = imgs[0][0]->size1;
	int img_w = imgs[0][0]->size2;
	int win_h = filters[0][0]->size1;
	int win_w = filters[0][0]->size2;

	if (!(n_channels == n_channels_img))
	{
		printf("ERROR: Mismatch in Number of Channels\n");
		return NULL;
	}

	int out_h = (img_h - win_h + 2 * pad_y) + 1;
	int out_w = (img_w - win_w + 2 * pad_x) + 1;

	// Create output array and Prepare padded image. Then perform convolution
	gsl_matrix*** out = (gsl_matrix ***) malloc(batch_size * sizeof(gsl_matrix**));
	for (int b = 0; b < batch_size; b++)
	{
		gsl_matrix** imgs_pad = (gsl_matrix**) malloc(n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < n_channels; c++) imgs_pad[c] = img_padding(imgs[b][c], pad_y, pad_x);

		out[b] = (gsl_matrix**) malloc(n_filters * sizeof(gsl_matrix*));
		for (int f = 0; f < n_filters; f++)
		{
			out[b][f] = gsl_matrix_calloc(out_h, out_w);
			for (int c = 0; c < n_channels; c++)
			{
				//gsl_matrix* conv_aux = conv2D(imgs_pad[c], filters[f][c], 1);		// True Convolutional
				gsl_matrix* conv_aux = conv2D_fft(imgs_pad[c], filters[f][c], 1);	// FFT Approximation
				gsl_matrix_add(out[b][f], conv_aux);
				gsl_matrix_free(conv_aux);
			}
		}

		for (int c = 0; c < n_channels; c++) gsl_matrix_free(imgs_pad[c]);
		free(imgs_pad);
	}

	return out;
}

// This function performs Forward Propagation
//    param x : Array of shape (batch_size, n_channels_img, img_height, img_width).
//              n_channels_img must match conv->n_channels
//    return  : Array of shape (batch_size, n_filters, out_height, out_width)
//    updates : conv_layer
gsl_matrix*** forward_conv(CONV* conv, gsl_matrix*** x)
{
	// Save "x" for back-propagation
	replace_image(&(conv->img), &x, conv->batch_size, conv->n_channels);

	// Performs convolution
	gsl_matrix*** y = conv_op(x, conv->batch_size, conv->n_channels,
		conv->W, conv->n_filters, conv->n_channels, conv->pad_y, conv->pad_x);

	for (int b = 0; b < conv->batch_size; b++)
		for (int f = 0; f < conv->n_filters; f++)
			gsl_matrix_add_constant(y[b][f], gsl_vector_get(conv->b, f));

	return y;
}

// This function performs Backward Propagation
//    param dy : Array of shape (batch_size, n_filters, out_height, out_width)
//    return   : Array of shape (batch_size, n_channels_img, img_height, img_width)
//    updates  : conv_layer
gsl_matrix*** backward_conv (CONV* conv, gsl_matrix*** dy)
{
        // Flip weights & Transpose channel/filter dimensions of weights
	gsl_matrix*** waux = (gsl_matrix***) malloc(conv->n_channels * sizeof(gsl_matrix**));
	for (int c = 0; c < conv->n_channels; c++)
	{
		waux[c] = (gsl_matrix**) malloc(conv->n_filters * sizeof(gsl_matrix*));
		for (int f = 0; f < conv->n_filters; f++)
		{
			waux[c][f] = gsl_matrix_alloc(conv->W[0][0]->size1, conv->W[0][0]->size2);
			gsl_matrix_memcpy(waux[c][f], conv->W[f][c]);
			for (int h = 0; h < conv->win_h/2; h++)
				gsl_matrix_swap_rows(waux[c][f], h, conv->win_h - h - 1);
			for (int w = 0; w < conv->win_w/2; w++)
				gsl_matrix_swap_columns(waux[c][f], w, conv->win_w - w - 1);
		}
	}

	// Propagate gradients to x
	gsl_matrix*** dx = conv_op(dy, conv->batch_size, conv->n_filters,
		waux, conv->n_channels, conv->n_filters, conv->pad_y, conv->pad_x);

	// Propagates gradients to weights and bias
	gsl_matrix*** grad_W = (gsl_matrix***) malloc(conv->n_filters * sizeof(gsl_matrix**));
	for (int f = 0; f < conv->n_filters; f++)
	{
		grad_W[f] = (gsl_matrix**) malloc(conv->n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < conv->n_channels; c++)
			grad_W[f][c] = gsl_matrix_calloc(conv->win_h, conv->win_w);
	}
	gsl_vector* grad_b = gsl_vector_calloc(conv->n_filters);

	int dy_h = dy[0][0]->size1;
	int dy_w = dy[0][0]->size2;
	for (int b = 0; b < conv->batch_size; b++)
	{
		// Prepares padded image
		gsl_matrix** x_pad = (gsl_matrix**) malloc(conv->n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < conv->n_channels; c++)
			x_pad[c] = img_padding(conv->img[b][c], conv->pad_y, conv->pad_x);

		// Propagates weights and bias
		for (int f = 0; f < conv->n_filters; f++)
		{
			// Propagate Weights
			for (int c = 0; c < conv->n_channels; c++)
			{
				gsl_matrix* conv_aux = conv2D(x_pad[c], dy[b][f], 1);
				gsl_matrix_add(grad_W[f][c], conv_aux);
				gsl_matrix_free(conv_aux);
			}

			// Propagate Bias
			double acc = gsl_vector_get(grad_b, f) + matrix_sum(dy[b][f]);
			gsl_vector_set(grad_b, f, acc);
		}

		for (int c = 0; c < conv->n_channels; c++)
			gsl_matrix_free(x_pad[c]);
		free(x_pad);
	}

	// Upgrade gradient values
	for (int f = 0; f < conv->n_filters; f++)
	{
		for (int c = 0; c < conv->n_channels; c++)
		{
			// Flip back grad_W frames
			for (int h = 0; h < conv->win_h/2; h++)
				gsl_matrix_swap_rows(grad_W[f][c], h, conv->win_h - h - 1);
			for (int w = 0; w < conv->win_w/2; w++)
				gsl_matrix_swap_columns(grad_W[f][c], w, conv->win_w - w - 1);

			// Update gradient values
			gsl_matrix_memcpy(conv->grad_W[f][c], grad_W[f][c]);

			// Free the local gradient frame
			gsl_matrix_free(grad_W[f][c]);
		}
		free(grad_W[f]);
	}
	free(grad_W);

	gsl_vector_memcpy(conv->grad_b, grad_b);
	gsl_vector_free(grad_b);

	// Free other auxiliar structures
	for (int c = 0; c < conv->n_channels; c++)
	{
		for (int f = 0; f < conv->n_filters; f++) gsl_matrix_free(waux[c][f]);
		free(waux[c]);
	}
	free(waux);

	return dx;
}

// Updates the Convolutional Layer
void get_updates_conv (CONV* conv, double lr)
{
	int size1 = conv->W[0][0]->size1;
	int size2 = conv->W[0][0]->size2;

	gsl_matrix* identity = gsl_matrix_calloc(size2, size2);
	gsl_matrix_set_all(identity, 1.0);
	gsl_matrix_set_identity(identity);

	double learn_factor = -1.0 * lr / conv->batch_size;
	for (int f = 0; f < conv->n_filters; f++)
		for (int c = 0; c < conv->n_channels; c++)
			gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, learn_factor, conv->grad_W[f][c], identity, 1.0, conv->W[f][c]);
	gsl_blas_daxpy(learn_factor, conv->grad_b, conv->b);

	gsl_matrix_free(identity);
}

// Initializes a convolutional layer
void create_CONV (CONV* conv, int n_channels, int n_filters, int filter_size,
	double scale, int border_mode, int batch_size)
{
	conv->batch_size = batch_size;
	conv->filter_size = filter_size;

	conv->n_filters = n_filters;
	conv->n_channels = n_channels;
	conv->win_h = filter_size;
	conv->win_w = filter_size;

	conv->W = (gsl_matrix***) malloc(n_filters * sizeof(gsl_matrix**));
	conv->grad_W = (gsl_matrix***) malloc(n_filters * sizeof(gsl_matrix**));
	for (int f = 0; f < n_filters; f++)
	{
		conv->W[f] = (gsl_matrix**) malloc(n_channels * sizeof(gsl_matrix*));
		conv->grad_W[f] = (gsl_matrix**) malloc(n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < n_channels; c++)
		{
			conv->W[f][c] = matrix_normal(filter_size, filter_size, 0, 1, scale);
			conv->grad_W[f][c] = gsl_matrix_calloc(filter_size, filter_size);
		}
	}
	conv->b = gsl_vector_calloc(n_filters);
	conv->grad_b = gsl_vector_calloc(n_filters);

	int padding = 0;
	if (border_mode == 1) padding = 0; 			// 'valid'
	if (border_mode == 2) padding = filter_size / 2;	// 'same'
	if (border_mode == 3) padding = filter_size - 1;	// 'full'

	conv->pad_y = padding;
	conv->pad_x = padding;

	conv->img = (gsl_matrix***) malloc(conv->batch_size * sizeof(gsl_matrix**));
	for (int b = 0; b < conv->batch_size; b++)
	{
		conv->img[b] = (gsl_matrix**) malloc(conv->n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < conv->n_channels; c++)
			conv->img[b][c] = gsl_matrix_calloc(1, 1);
	}
}

// Destructor of Convolutional Layer
void free_CONV (CONV* conv)
{
	// Free last batch of images
	for (int b = 0; b < conv->batch_size; b++)
	{
		for (int ci = 0; ci < conv->n_channels; ci++) gsl_matrix_free(conv->img[b][ci]);
		free(conv->img[b]);
	}
	free(conv->img);

	// Free weights matrix and bias
	for (int f = 0; f < conv->n_filters; f++)
	{
		for (int ci = 0; ci < conv->n_channels; ci++)
		{
			gsl_matrix_free(conv->W[f][ci]);
			gsl_matrix_free(conv->grad_W[f][ci]);
		}
		free(conv->W[f]);
		free(conv->grad_W[f]);
	}
	free(conv->W);
	free(conv->grad_W);

	gsl_vector_free(conv->b);
	gsl_vector_free(conv->grad_b);
}

// Function to copy a CONV layer
// Important: destination must NOT be initialized
void copy_CONV (CONV* destination, CONV* origin)
{
	destination->batch_size = origin->batch_size;
	destination->filter_size = origin->filter_size;

	destination->n_filters = origin->n_filters;
	destination->n_channels = origin->n_channels;
	destination->win_h = origin->win_h;
	destination->win_w = origin->win_w;

	destination->pad_y = origin->pad_y;
	destination->pad_x = origin->pad_x;

	destination->W = (gsl_matrix***) malloc(origin->n_filters * sizeof(gsl_matrix**));
	destination->grad_W = (gsl_matrix***) malloc(origin->n_filters * sizeof(gsl_matrix**));
	for (int f = 0; f < origin->n_filters; f++)
	{
		destination->W[f] = (gsl_matrix**) malloc(origin->n_channels * sizeof(gsl_matrix*));
		destination->grad_W[f] = (gsl_matrix**) malloc(origin->n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < origin->n_channels; c++)
		{
			destination->W[f][c] = gsl_matrix_alloc(origin->win_h, origin->win_w);
			destination->grad_W[f][c] = gsl_matrix_alloc(origin->win_h, origin->win_w);
			gsl_matrix_memcpy(destination->W[f][c], origin->W[f][c]);
			gsl_matrix_memcpy(destination->grad_W[f][c], origin->grad_W[f][c]);
		}
	}

	destination->b = gsl_vector_calloc(origin->n_filters);
	destination->grad_b = gsl_vector_calloc(origin->n_filters);
	gsl_vector_memcpy(destination->b, origin->b);
	gsl_vector_memcpy(destination->grad_b, origin->grad_b);

	int img_h = origin->img[0][0]->size1;
	int img_w = origin->img[0][0]->size2;

	destination->img = (gsl_matrix***) malloc(origin->batch_size * sizeof(gsl_matrix**));
	for (int b = 0; b < origin->batch_size; b++)
	{
		destination->img[b] = (gsl_matrix**) malloc(origin->n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < origin->n_channels; c++)
		{
			destination->img[b][c] = gsl_matrix_alloc(img_h, img_w);
			gsl_matrix_memcpy(destination->img[b][c], origin->img[b][c]);
		}
	}
}

// Function to compare a CONV layer
int compare_CONV (CONV* C1, CONV* C2)
{
	int equal = 1;

	if (
		C1->batch_size != C2->batch_size ||
		C1->filter_size != C2->filter_size ||
		C1->n_filters != C2->n_filters ||
		C1->n_channels != C2->n_channels ||
		C1->win_h != C2->win_h ||
		C1->win_w != C2->win_w ||
		C1->pad_y != C2->pad_y ||
		C1->pad_x != C2->pad_x
	) equal = 0;

	for (int f = 0; f < C2->n_filters; f++)
		for (int c = 0; c < C2->n_channels; c++)
		{
			equal = equal * gsl_matrix_equal(C1->W[f][c], C2->W[f][c]);
			equal = equal * gsl_matrix_equal(C1->grad_W[f][c], C2->grad_W[f][c]);
		}
	equal = equal * gsl_vector_equal(C1->b, C2->b);
	equal = equal * gsl_vector_equal(C1->grad_b, C2->grad_b);

	for (int b = 0; b < C2->batch_size; b++)
		for (int c = 0; c < C2->n_channels; c++)
			equal = equal * gsl_matrix_equal(C1->img[b][c], C2->img[b][c]);

	return equal;
}
