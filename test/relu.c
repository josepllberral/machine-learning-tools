/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* RELU ACTIVATION LAYERS                                                    */
/*---------------------------------------------------------------------------*/

// Forwards x by setting max_0
//	param x :	Array
//	returns :	Array applied max_0
//	updates :	relu_layer
gsl_matrix*** forward_relu (RELU* relu, gsl_matrix*** x)
{
	int img_h = x[0][0]->size1;
	int img_w = x[0][0]->size2;

	// Process image
	gsl_matrix*** out = (gsl_matrix***) malloc(relu->batch_size * sizeof(gsl_matrix**));
	for (int b = 0; b < relu->batch_size; b++)
	{
		out[b] = (gsl_matrix**) malloc(relu->n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < relu->n_channels; c++)
		{
			out[b][c] = gsl_matrix_calloc(img_h, img_w);
			gsl_matrix_memcpy(out[b][c], x[b][c]);
			for (int h = 0; h < img_h; h++)
				for (int w = 0; w < img_w; w++)
					if (gsl_matrix_get(out[b][c], h, w) < 0)
						gsl_matrix_set(out[b][c], h, w, 0);
		}
	}

	// Save modified image in relu
	replace_image(&(relu->img), &out, relu->batch_size, relu->n_channels);
	
	return out;
}

// Returns a value activated
//	param dy :	Array
//	return   :	Array passed through (max_0)
//	updates  :	relu_layer (does nothing)
gsl_matrix*** backward_relu (RELU* relu, gsl_matrix*** dy)
{
	int img_h = dy[0][0]->size1;
	int img_w = dy[0][0]->size2;

	// 'De-Process' image
	gsl_matrix*** out = (gsl_matrix***) malloc(relu->batch_size * sizeof(gsl_matrix**));
	for (int b = 0; b < relu->batch_size; b++)
	{
		out[b] = (gsl_matrix**) malloc(relu->n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < relu->n_channels; c++)
		{
			out[b][c] = gsl_matrix_calloc(img_h, img_w);
			gsl_matrix_memcpy(out[b][c], dy[b][c]);
			for (int h = 0; h < img_h; h++)
				for (int w = 0; w < img_w; w++)
					if (gsl_matrix_get(out[b][c], h, w) < 0)
						gsl_matrix_set(out[b][c], h, w, 0); // FIXME - Check in the future...
		}
	}

	return out;
}

// Updates the ReLU Layer (Does Nothing)
void get_updates_relu (RELU* relu, double lr)
{
	return;
}


// Initializes a ReLU layer
void create_RELU (RELU* relu, int n_channels, int batch_size)
{
	relu->batch_size = batch_size;
	relu->n_channels = n_channels;

	relu->img = (gsl_matrix***) malloc(batch_size * sizeof(gsl_matrix**));
	for (int b = 0; b < batch_size; b++)
	{
		relu->img[b] = (gsl_matrix**) malloc(n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < n_channels; c++)
			relu->img[b][c] = gsl_matrix_calloc(1, 1);
	}
}

// Destructor of ReLU Layer
void free_RELU (RELU* relu)
{
	// Free last batch of images
	for (int b = 0; b < relu->batch_size; b++)
	{
		for (int c = 0; c < relu->n_channels; c++) gsl_matrix_free(relu->img[b][c]);
		free(relu->img[b]);
	}
	free(relu->img);

	return;
}


// Function to copy a ReLU layer
// Important: destination must NOT be initialized
void copy_RELU (RELU* destination, RELU* origin)
{
	destination->batch_size = origin->batch_size;
	destination->n_channels = origin->n_channels;

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

// Function to compare a ReLU layer
int compare_RELU (RELU* C1, RELU* C2)
{
	int equal = 1;

	if (
		C1->batch_size != C2->batch_size ||
		C1->n_channels != C2->n_channels
	) equal = 0;

	for (int b = 0; b < C2->batch_size; b++)
		for (int c = 0; c < C2->n_channels; c++)
			equal = equal * gsl_matrix_equal(C1->img[b][c], C2->img[b][c]);

	return equal;
}

