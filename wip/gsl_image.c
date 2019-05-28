/*----------------------------------------------------------------------------*/
/* gsl_image: 3D matrix operations                                            */
/*----------------------------------------------------------------------------*/

typedef struct
{
	int size1;
	int size2;
	int size3;
	gsl_matrix** matrix;
} gsl_image;

gsl_image* gsl_image_calloc(int size1, int size2, int size3)
{
	gsl_matrix** M = (gsl_matrix **) malloc(size1 * sizeof(gsl_matrix*));
	for (int i = 0; i < size1; i++)	M[i] = gsl_matrix_calloc(size2, size3);
	
	gsl_image* retval;
	retval->dim1 = dim1;
	retval->dim2 = dim2;
	retval->dim3 = dim3;
	retval->matrix = M;

	return retval;
}

void gsl_image_free(gsl_image* image)
{
	for (int i = 0; i < image->size1; i++)
		gsl_matrix_free(image->matrix[i]);
	free(image->matrix);
}

void gsl_image_memcpy(gsl_image* destination, gsl_image* origin)
{
	destination->size1 = origin->size1;
	destination->size2 = origin->size2;
	destination->size3 = origin->size3;

	destination->matrix = (gsl_matrix**) malloc(origin->size1 * sizeof(gsl_matrix*));
	for (int b = 0; b < origin->size1; b++)
	{
		destination->matrix[b] = gsl_matrix_alloc(origin->size2, origin->size3);
		gsl_matrix_memcpy(destination->matrix[b], origin->matrix[b]);
	}
}

void gsl_image_setall(gsl_image* target, double d)
{
	for (int i = 0; i < target->size1; i++)
		gsl_matrix_set_all(target[i], d);
}

int gsl_image_equal(gsl_image* C1, gsl_image* C2)
{
	int equal = 1;

	if (
		C1->size1 != C2->size1 ||
		C1->size2 != C2->size2 ||
		C1->size3 != C2->size3
	) equal = 0;

	for (int a = 0; a < C2->size1; a++)
		equal = equal * gsl_matrix_equal(C1->matrix[a], C2->matrix[a]);

	return equal;
}

/*----------------------------------------------------------------------------*/
/* gsl_batch: 4D matrix operations                                            */
/*----------------------------------------------------------------------------*/

typedef struct
{
	int size1;
	int size2;
	int size3;
	int size4;
	gsl_image** image;
} gsl_batch;

gsl_batch* gsl_batch_calloc(int size1, int size2, int size3, int size4)
{
	gsl_image** I = (gsl_image **) malloc(size1 * sizeof(gsl_image*));
	for (int i = 0; i < size1; i++) I[i] = gsl_image_calloc(size2, size3, size4);
	
	gsl_batch* retval;
	retval->dim1 = dim1;
	retval->dim2 = dim2;
	retval->dim3 = dim3;
	retval->dim4 = dim4;
	retval->image = I;

	return retval;
}

void gsl_batch_free(gsl_batch* batch)
{
	for (int i = 0; i < batch->size1; i++) gsl_image_free(batch->image[i]);
	free(batch->image);
}

void gsl_batch_memcpy(gsl_batch* destination, gsl_batch* origin)
{
	destination->size1 = origin->size1;
	destination->size2 = origin->size2;
	destination->size3 = origin->size3;
	destination->size4 = origin->size4;

	destination->image = (gsl_image *) malloc(origin->size1 * sizeof(gsl_image));
	for (int i = 0; i < origin->size1; i++)
		gsl_image_memcpy(destination->image[i], origin->image[i]);
}

void gsl_batch_set_all(gsl_batch* target, double d)
{
	for (int i = 0; i < target->size1; i++)
		gsl_image_set_all(target->image[i], d);
}

int gsl_batch_equal(gsl_batch* C1, gsl_batch* C2)
{
	int equal = 1;

	if (
		C1->size1 != C2->size1 ||
		C1->size2 != C2->size2 ||
		C1->size3 != C2->size3 ||
		C1->size4 != C2->size4
	) equal = 0;

	for (int a = 0; a < C2->size1; a++)
		equal = equal * gsl_image_equal(C1->image[a], C2->image[a]);

	return equal;
}
