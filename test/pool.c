/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations
// Compile using "gcc cnn.c conv.c pool.c flat.c grad_check.c matrix_ops.c -lgsl -lgslcblas -lm -o cnn"

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* POOLING LAYERS                                                            */
/*---------------------------------------------------------------------------*/

// Forwards a Pooling Matrix (4D) from a Convolutional Matrix (4D)
//	param x :	Array of shape (batch_size, n_channels, img_height, img_width)
//	returns :	Array of shape (batch_size, n_channels, out_height, out_width)
//	updates :	pool_layer
gsl_matrix*** forward_pool(POOL* pool, gsl_matrix*** imgs)
{
	// Save "imgs" for back-propagation
	replace_image(&(pool->img), &imgs, pool->batch_size, pool->n_channels);

	// Create output array and Prepare padded image for convolution
	int img_h = imgs[0][0]->size1;
	int img_w = imgs[0][0]->size2;

	int out_h = (img_h - pool->win_size + 2 * pool->padding) / pool->stride + 1;
	int out_w = (img_w - pool->win_size + 2 * pool->padding) / pool->stride + 1;

	gsl_matrix*** out = (gsl_matrix ***) malloc(pool->batch_size * sizeof(gsl_matrix**));
	for (int i = 0; i < pool->batch_size; i++)
	{
		out[i] = (gsl_matrix**) malloc(pool->n_channels * sizeof(gsl_matrix*));
		for (int j = 0; j < pool->n_channels; j++)
			out[i][j] = gsl_matrix_calloc(out_h, out_w);
	}

	// Perform average pooling
	double ws2 = 1 / (pool->win_size * pool->win_size);
	for (int b = 0; b < pool->batch_size; b++)
		for (int c = 0; c < pool->n_channels; c++)
		{
			gsl_matrix_scale(imgs[b][c], ws2);
			for (int h = 0; h < out_h; h++)
			{
				int yaux = h * pool->stride - 1;
				int lim_y1 = max(yaux, 0);
				int lim_y2 = min((yaux + pool->win_size - 1), img_h);

				for (int w = 0; w < out_w; w++)
				{
					int xaux = w * pool->stride - 1;
					int lim_x1 = max(xaux, 0);
					int lim_x2 = min((xaux + pool->win_size - 1), img_w);

					double acc = 0;
					for (int y = lim_y1; y < lim_y2; y++)
						for (int x = lim_x1; x < lim_x2; x++)					
							acc += gsl_matrix_get(imgs[b][c], y, x);

					gsl_matrix_set(out[b][c], h, w, acc);
				}
			}
		}
	return out;
}

// Backwards a Pooling Matrix (4D) to a Convolutional Matrix (4D)
//	param dy :	Array of shape (batch_size, n_channels, out_height, out_width)
//	return   :	Array of shape (batch_size, n_channels, img_height, img_width)
//	updates  :	pool_layer
gsl_matrix*** backward_pool(POOL* pool, gsl_matrix*** dy)
{
	int dx_h = pool->img[0][0]->size1;
	int dx_w = pool->img[0][0]->size2;

	int dy_h = dy[0][0]->size1;
	int dy_w = dy[0][0]->size2;

	// Un-do pooling
	gsl_matrix*** dx = (gsl_matrix ***) malloc(pool->batch_size * sizeof(gsl_matrix**));
	double ws2 = 1 / (pool->win_size * pool->win_size);
	for (int b = 0; b < pool->batch_size; b++)
	{
		dx[b] = (gsl_matrix**) malloc(pool->n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < pool->n_channels; c++)
		{
			gsl_matrix_scale(dy[b][c], ws2);
			dx[b][c] = gsl_matrix_calloc(dx_h, dx_w);
			for (int h = 0; h < dy_h; h++)
			{
				int yaux = h * pool->stride - 1;
				int lim_y1 = max(yaux, 0);
				int lim_y2 = min(yaux + pool->win_size - 1, dx_h);
				for (int w = 0; w < dy_w; w++)
				{
					double dyval = gsl_matrix_get(dy[b][c], h, w);

					int xaux = w * pool->stride - 1;
					int lim_x1 = max(xaux, 0);
					int lim_x2 = min(xaux + pool->win_size - 1, dx_w);

					for (int y = lim_y1; y < lim_y2; y++)
						for (int x = lim_x1; x < lim_x2; x++)
							gsl_matrix_set(dx[b][c], y, x, gsl_matrix_get(dx[b][c], y, x) + dyval);
				}
			}
		}
	}

	return dx;
}

// Updates the Pooling Layer (Does Nothing...)
void get_updates_pool (POOL* pool, double lr)
{
	return;
}

// Initializes a Pooling Layer
void create_POOL (POOL* pool, int n_channels, double scale, int batch_size, int win_size, int stride)
{
	pool->batch_size = batch_size;
	pool->n_channels = n_channels;
	pool->win_size = win_size;
	pool->padding = win_size / 2;
	pool->stride = stride;

	pool->img = (gsl_matrix***) malloc(pool->batch_size * sizeof(gsl_matrix**));
	for (int b = 0; b < pool->batch_size; b++)
	{
		pool->img[b] = (gsl_matrix**) malloc(pool->n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < pool->n_channels; c++)
			pool->img[b][c] = gsl_matrix_calloc(1, 1);
	}
}

// Destructor of Pooling Layer
void free_POOL (POOL* pool)
{
	// Free last batch of images
	for (int b = 0; b < pool->batch_size; b++)
	{
		for (int c = 0; c < pool->n_channels; c++) gsl_matrix_free(pool->img[b][c]);
		free(pool->img[b]);
	}
	free(pool->img);
}

// Function to copy a POOL layer
void copy_POOL (POOL* destination, POOL* origin)
{
	destination->batch_size = origin->batch_size;
	destination->n_channels = origin->n_channels;
	destination->win_size = origin->win_size;
	destination->padding = origin->padding;
	destination->stride = origin->stride;

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

// Function to compare a POOL layer
int compare_POOL (POOL* C1, POOL* C2)
{
	int equal = 1;

	if (
		C1->batch_size != C2->batch_size ||
		C1->n_channels != C2->n_channels ||
		C1->win_size != C2->win_size ||
		C1->padding != C2->padding ||
		C1->stride != C2->stride
	) equal = 0;

	int img_h = C2->img[0][0]->size1;
	int img_w = C2->img[0][0]->size2;
	for (int b = 0; b < C2->batch_size; b++)
		for (int c = 0; c < C2->n_channels; c++)
			for (int h = 0; h < img_h; h++)
				for (int w = 0; w < img_w; w++)
				{
					double i = gsl_matrix_get(C1->img[b][c], h, w);
					double j = gsl_matrix_get(C2->img[b][c], h, w);
					if (i != j) equal = 0;
				}

	return equal;
}

