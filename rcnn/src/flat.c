/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* FLATTENING LAYERS                                                         */
/*---------------------------------------------------------------------------*/

// Creates a Flat Vector (2D) from a Convolutional Matrix (4D)
//	param x :	Array of shape (batch_size, n_channels, img_height, img_width)
//	returns :	Array of shape (batch_size, n_channels * img_height * img_width)
//	updates :	flat_layer
gsl_matrix* forward_flat (FLAT* flat, gsl_matrix*** x)
{
	flat->img_h = x[0][0]->size1;
 	flat->img_w = x[0][0]->size2;

	gsl_matrix* out = gsl_matrix_alloc(flat->batch_size, flat->n_channels * flat->img_h * flat->img_w);
	for (int b = 0; b < flat->batch_size; b++)
		for (int c = 0; c < flat->n_channels; c++)
			for (int h = 0; h < flat->img_h; h++)
				for (int w = 0; w < flat->img_w; w++)
				{
					int idx = w + h * flat->img_w + c * flat->img_h * flat->img_w;
					double value = gsl_matrix_get(x[b][c], h, w);
					gsl_matrix_set(out, b, idx, value);
				}
	return out;
}

// Unflattens a Flat Vector (2D) to a Convolutional Matrix (4D)
//	param dy :	Array of shape (batch_size, n_channels * img_height * img_width)
//	return   :	Array of shape (batch_size, n_channels, img_height, img_width)
//	updates  :	flat_layer (does nothing)
gsl_matrix*** backward_flat (FLAT* flat, gsl_matrix* dy)
{
	gsl_matrix*** out = (gsl_matrix***) malloc(flat->batch_size * sizeof(gsl_matrix**));
	for (int b = 0; b < flat->batch_size; b++)
	{
		out[b] = (gsl_matrix**) malloc(flat->n_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < flat->n_channels; c++)
		{
			out[b][c] = gsl_matrix_calloc(flat->img_h, flat->img_w);
			for (int p = 0; p < flat->img_h * flat->img_w; p++)
			{
				int idx = p + (flat->img_h * flat->img_w) * c;
				double value = gsl_matrix_get(dy, b, idx);

				int p_x = p / flat->img_h;
				int p_y = p % flat->img_h;

				gsl_matrix_set(out[b][c], p_x, p_y, value);
			}
		}
	}

	return out;
}

// Updates the Flat Layer (Does Nothing)
void get_updates_flat (FLAT* flat, double lr)
{
	return;
}

// Initializes a Flattening layer
void create_FLAT (FLAT* flat, int n_channels, int batch_size)
{
	flat->batch_size = batch_size;
	flat->n_channels = n_channels;
	flat->img_h = 0;
	flat->img_w = 0;
}

// Destructor of Flattening Layer (Does Nothing)
void free_FLAT (FLAT* flat)
{
	return;
}


// Function to copy a Flat layer
void copy_FLAT (FLAT* destination, FLAT* origin)
{
	destination->batch_size = origin->batch_size;
	destination->n_channels = origin->n_channels;
	destination->img_h = origin->img_h;
	destination->img_w = origin->img_w;
}

// Function to compare a FLAT layer
int compare_FLAT (FLAT* C1, FLAT* C2)
{
	int equal = 1;

	if (
		C1->batch_size != C2->batch_size ||
		C1->n_channels != C2->n_channels ||
		C1->img_h != C2->img_h ||
		C1->img_w != C2->img_w
	) equal = 0;

	return equal;
}

