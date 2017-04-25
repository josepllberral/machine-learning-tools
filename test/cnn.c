/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations
// Compile using "gcc cnn.c conv.c pool.c flat.c relu.c line.c soft.c cell.c grad_check.c matrix_ops.c -lgsl -lgslcblas -lm -o cnn"

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* AUXILIAR FUNCTIONS                                                        */
/*---------------------------------------------------------------------------*/

// Function to binarize a vector
//  param vec    : vector of integers (factors and strings must be numerized before)
//  param vlen   : length of vector
//  param vclass : number of classes in vec => lenght(unique(vec))
/*int** binarization_cnn(int* vec, int vlen, int vclass)
{
	int** out = (int**) malloc(sizeof(int*) * vlen);
	for (int i = 0; i < vlen; i++)
	{
		out[i] = (int*) calloc(vclass, sizeof(int));
		out[i][vec[i]] = 1;
	}

	return out;
}*/

void replace_image(gsl_matrix**** destination, gsl_matrix**** newimage, int size1, int size2)
{
	for (int b = 0; b < size1; b++)
	{
		for (int c = 0; c < size2; c++) gsl_matrix_free((*destination)[b][c]);
		free((*destination)[b]);
	}
	free((*destination));

	int img_h = (*newimage)[0][0]->size1;
	int img_w = (*newimage)[0][0]->size2;

	(*destination) = (gsl_matrix***) malloc(size1 * sizeof(gsl_matrix**));
	for (int b = 0; b < size1; b++)
	{
		(*destination)[b] = (gsl_matrix**) malloc(size2 * sizeof(gsl_matrix*));
		for (int c = 0; c < size2; c++)
		{
			(*destination)[b][c] = gsl_matrix_alloc(img_h, img_w);
			gsl_matrix_memcpy((*destination)[b][c], (*newimage)[b][c]);
		}
	}
}

/*---------------------------------------------------------------------------*/
/* HOW TO TRAIN YOUR CNN                                                     */
/*---------------------------------------------------------------------------*/

// TODO - ...

/*---------------------------------------------------------------------------*/
/* DRIVERS FOR TEST                                                          */
/*---------------------------------------------------------------------------*/

// Driver for Convolutional Layer
int main_conv()
{
	int batch_size = 10;
	int n_channels = 1;	// Will and must be the same in image and filters
	int img_shape_h = 5;
	int img_shape_w = 5;
	int n_filters = 2;
	int filter_size = 3;

	int border_mode = 2;	// 1 = 'valid', 2 = 'same'

	// Create random image
	gsl_matrix*** x = (gsl_matrix***) malloc(batch_size * sizeof(gsl_matrix**));
	for (int b = 0; b < batch_size; b++)
	{
		x[b] = (gsl_matrix**) malloc(n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < n_channels; c++)
			x[b][c] = matrix_normal(img_shape_h, img_shape_w, 0, 1, 10);
	}

	printf("Create Convolution Layer\n");

	CONV conv;
	create_CONV(&conv, n_channels, n_filters, filter_size, 0.01, border_mode, batch_size);

	printf("Initialize gradients\n");

	// Initialize just for gradient check
	for (int b = 0; b < batch_size; b++)
		for (int c = 0; c < n_channels; c++)
		{
			gsl_matrix_free(conv.img[b][c]);
			conv.img[b][c] = gsl_matrix_calloc(img_shape_h, img_shape_w);
		}

	printf("Start Gradient Check\n");

	// Gradient check
	int a = check_grad_conv(&conv, x, 1234, -1, -1, -1);
	if (a == 0) printf("Gradient check passed\n");

	printf("Fin Gradient Check\n");

	for (int b = 0; b < batch_size; b++)
	{
		for (int c = 0; c < n_channels; c++) gsl_matrix_free(x[b][c]);
		free(x[b]);
	}
	free(x);

	free_CONV(&conv);

	return 0;
}

// Driver for Pooling Layer
int main_pool()
{
	int batch_size = 1;
	int n_channels = 1;
	int img_shape_h = 5;
	int img_shape_w = 5;
	int win_size = 3;
	int stride = 2;

	// Create random image
	gsl_matrix*** x = (gsl_matrix***) malloc(batch_size * sizeof(gsl_matrix**));
	for (int b = 0; b < batch_size; b++)
	{
		x[b] = (gsl_matrix**) malloc(n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < n_channels; c++)
			x[b][c] = matrix_normal(img_shape_h, img_shape_w, 0, 1, 10);
	}

	printf("Create Pooling Layer\n");

	POOL pool;
	create_POOL(&pool, n_channels, 0.01, batch_size, win_size, stride);

	printf("Initialize gradients\n");

	// Initialize just for gradient check
	for (int b = 0; b < batch_size; b++)
		for (int c = 0; c < n_channels; c++)
		{
			gsl_matrix_free(pool.img[b][c]);
			pool.img[b][c] = gsl_matrix_calloc(img_shape_h, img_shape_w);
		}

	printf("Start Gradient Check\n");

	// Gradient check
	int a = check_grad_pool(&pool, x, 1234, -1, -1, -1);
	if (a == 0) printf("Gradient check passed\n");

	printf("Fin Gradient Check\n");

	for (int b = 0; b < batch_size; b++)
	{
		for (int c = 0; c < n_channels; c++) gsl_matrix_free(x[b][c]);
		free(x[b]);
	}
	free(x);

	free_POOL(&pool);

	return 0;
}

// Driver for Flattening Layer
int main_flat()
{
	int batch_size = 2;
	int n_channels = 1;
	int img_shape_h = 5;
	int img_shape_w = 5;

	// Create random image
	gsl_matrix*** x = (gsl_matrix***) malloc(batch_size * sizeof(gsl_matrix**));
	for (int b = 0; b < batch_size; b++)
	{
		x[b] = (gsl_matrix**) malloc(n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < n_channels; c++)
			x[b][c] = matrix_normal(img_shape_h, img_shape_w, 0, 1, 10);
	}

	printf("Create Flattening Layer\n");

	FLAT flat;
	create_FLAT(&flat, n_channels, batch_size);

	printf("Start Gradient Check\n");

	// Gradient check
	int a = check_grad_flat(&flat, x, 1234, -1, -1, -1);
	if (a == 0) printf("Gradient check passed\n");

	printf("Fin Gradient Check\n");

	for (int b = 0; b < batch_size; b++)
	{
		for (int c = 0; c < n_channels; c++) gsl_matrix_free(x[b][c]);
		free(x[b]);
	}
	free(x);

	free_FLAT(&flat);

	return 0;
}

// Driver for ReLU Layer
int main_relu()
{
	int batch_size = 2;
	int n_channels = 1;
	int img_shape_h = 5;
	int img_shape_w = 5;

	// Create random image
	gsl_matrix*** x = (gsl_matrix***) malloc(batch_size * sizeof(gsl_matrix**));
	for (int b = 0; b < batch_size; b++)
	{
		x[b] = (gsl_matrix**) malloc(n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < n_channels; c++)
			x[b][c] = matrix_normal(img_shape_h, img_shape_w, 0, 1, 10);
	}

	printf("Create ReLU Layer\n");

	RELU relu;
	create_RELU(&relu, n_channels, batch_size);

	printf("Start Gradient Check\n");

	// Gradient check
	int a = check_grad_relu(&relu, x, 1234, -1, -1, -1);
	if (a == 0) printf("Gradient check passed\n");

	printf("Fin Gradient Check\n");

	for (int b = 0; b < batch_size; b++)
	{
		for (int c = 0; c < n_channels; c++) gsl_matrix_free(x[b][c]);
		free(x[b]);
	}
	free(x);

	free_RELU(&relu);

	return 0;
}

// Driver for Linear Layer
int main_line()
{
	int batch_size = 2;
	int n_visible = 10;
	int n_hidden = 5;

	// Create random input
	gsl_matrix* x = matrix_normal(batch_size, n_visible, 0, 1, 10);

	printf("Create Linear Layer\n");

	LINE line;
	create_LINE(&line, n_visible, n_hidden, 0.01, batch_size);

	printf("Start Gradient Check\n");

	// Gradient check
	int a = check_grad_line(&line, x, 1234, -1, -1, -1);
	if (a == 0) printf("Gradient check passed\n");

	printf("Fin Gradient Check\n");

	gsl_matrix_free(x);

	free_LINE(&line);

	return 0;
}

int main_soft()
{
	int batch_size = 2;
	int n_units = 10;

	// Create random input
	gsl_matrix* x = matrix_normal(batch_size, n_units, 0, 1, 10);

	printf("Create Soft Layer\n");

	SOFT soft;
	create_SOFT(&soft, n_units, batch_size);

	printf("Start Gradient Check\n");

	// Gradient check
	int a = check_grad_soft(&soft, x, 1234, -1, -1, -1);
	if (a == 0) printf("Gradient check passed\n");

	printf("Fin Gradient Check\n");

	gsl_matrix_free(x);

	free_SOFT(&soft);

	return 0;
}

int main_cell()
{
	int batch_size = 10;
	int n_units = 2;

	// Create random input and output
	gsl_matrix* x = matrix_normal(batch_size, n_units, 0, 1, 10);
	gsl_matrix* y = matrix_normal(batch_size, n_units, 0, 1, 10);

	for (int i = 0; i < batch_size; i++)
		for (int j = 0; j < n_units; j++)
		{
			gsl_matrix_set(x, i, j, abs(gsl_matrix_get(x, i, j)));
			gsl_matrix_set(y, i, j, abs(gsl_matrix_get(y, i, j)));
		}

	printf("Create Cross-Entropy Layer\n");

	CELL cell;
	create_CELL(&cell);

	printf("Start Gradient Check\n");

	// Gradient check
	int a = check_grad_cell(&cell, x, y, 1234, -1, -1, -1);
	if (a == 0) printf("Gradient check passed\n");

	printf("Fin Gradient Check\n");

	gsl_matrix_free(x);
	gsl_matrix_free(y);

	free_CELL(&cell);

	return 0;
}

/*---------------------------------------------------------------------------*/
/* MAIN FUNCTION - TEST                                                      */
/*---------------------------------------------------------------------------*/

int main()
{
	return main_cell();
}
