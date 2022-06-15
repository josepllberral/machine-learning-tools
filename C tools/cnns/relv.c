/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C                                               */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* RELU (SINGLE MATRIX VERSION) ACTIVATION LAYERS                            */
/*---------------------------------------------------------------------------*/

// Forwards x by setting max_0
//	param x :	Matrix
//	returns :	Matrix applied max_0
//	updates :	relv_layer
gsl_matrix* forward_relv (RELV* relv, gsl_matrix* x)
{
	int img_h = (int) x->size1;
	int img_w = (int) x->size2;

	// Process image
	gsl_matrix* out = gsl_matrix_alloc(img_h, img_w);
	gsl_matrix_memcpy(out, x);
	for (int h = 0; h < img_h; h++)
		for (int w = 0; w < img_w; w++)
			if (gsl_matrix_get(x, h, w) < 0)
				gsl_matrix_set(out, h, w, 0);

	// Save modified image in relv
	gsl_matrix_free(relv->img);
	relv->img = gsl_matrix_calloc(img_h, img_w);
	gsl_matrix_memcpy(relv->img, out);

	return out;
}

// Returns a value activated
//	param dy :	Matrix
//	return   :	Matrix passed through (max_0)
//	updates  :	relv_layer (does nothing)
gsl_matrix* backward_relv (RELV* relv, gsl_matrix* dy)
{
	int img_h = dy->size1;
	int img_w = dy->size2;

	// 'De-Process' image
	gsl_matrix* out = gsl_matrix_calloc(img_h, img_w);
	gsl_matrix_memcpy(out, dy);
	for (int h = 0; h < img_h; h++)
		for (int w = 0; w < img_w; w++)
			if (gsl_matrix_get(relv->img, h, w) < 0)
				gsl_matrix_set(out, h, w, 0);

	return out;
}

// Updates the ReLU-V Layer (Does Nothing)
void get_updates_relv (RELV* relu, double lr)
{
	return;
}

// Initializes a ReLU-V layer
void create_RELV (RELV* relv, int batch_size)
{
	relv->batch_size = batch_size;
	relv->img = gsl_matrix_calloc(1, 1);
}

// Destructor of ReLU-V Layer
void free_RELV (RELV* relv)
{
	// Free last batch of images
	gsl_matrix_free(relv->img);

	return;
}


// Function to copy a ReLU-V layer
// Important: destination must NOT be initialized
void copy_RELV (RELV* destination, RELV* origin)
{
	int img_h = origin->img->size1;
	int img_w = origin->img->size2;

	destination->img = gsl_matrix_alloc(img_h, img_w);
	gsl_matrix_memcpy(destination->img, origin->img);
}

// Function to compare a ReLU-V layer
int compare_RELV (RELV* C1, RELV* C2)
{
	int equal = 1;

	equal = equal * gsl_matrix_equal(C1->img, C2->img);

	return equal;
}
