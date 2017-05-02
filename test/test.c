/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Test Driver Implementations

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* DRIVERS FOR CNN TEST                                                      */
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

// Driver for Softmax Layer
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

// Driver for Cross-Entropy Layer
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

// Driver for Full CNN network using MNIST
int main_cnn()
{
	printf("Start\n");

	int nrow = 60000;
	int ncol = 784;

	int num_channels = 1;
	int img_h = 28;
	int img_w = 28;

	FILE * fp;
	char * line = NULL;
	size_t len = 0;

	// Read Train Data (MNIST)
	fp = fopen("../rrbm/datasets/mnist_trainx.data", "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	gsl_matrix*** training_x = (gsl_matrix***) malloc(nrow * sizeof(gsl_matrix**));
	for (int i = 0; i < nrow; i++)
	{
		training_x[i] = (gsl_matrix**) malloc(1 * sizeof(gsl_matrix*));
		// ...because num_channels = 1
		training_x[i][0] = gsl_matrix_alloc(img_h, img_w);

		ssize_t read = getline(&line, &len, fp);
		char *ch = strtok(line, " ");
		for (int j = 0; j < ncol; j++)
		{
			int idx_h = j % img_h;
			int idx_w = j / img_w;
			gsl_matrix_set(training_x[i][0], idx_h, idx_w, atof(ch));
			ch = strtok(NULL, " ");
		}
		free(ch);
	}
	fclose(fp);

	fp = fopen("../rrbm/datasets/mnist_trainy.data", "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	gsl_matrix* training_y = gsl_matrix_calloc(nrow, 10);
	for (int i = 0; i < nrow; i++)
	{
		ssize_t read = getline(&line, &len, fp);
		int y = atoi(line);
		gsl_matrix_set(training_y, i, y, 1.0);
	}
	fclose(fp);

	printf("Training Dataset Read\n");

	// Read Test Data (MNIST)
	fp = fopen("../rrbm/datasets/mnist_testx.data", "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	nrow = 10000;

	gsl_matrix*** testing_x = (gsl_matrix***) malloc(nrow * sizeof(gsl_matrix**));
	for (int i = 0; i < nrow; i++)
	{
		testing_x[i] = (gsl_matrix**) malloc(1 * sizeof(gsl_matrix*));
		// ...because num_channels = 1
		testing_x[i][0] = gsl_matrix_alloc(img_h, img_w);

		ssize_t read = getline(&line, &len, fp);
		char *ch = strtok(line, " ");
		for (int j = 0; j < ncol; j++)
		{
			int idx_h = j % img_h;
			int idx_w = j / img_w;
			gsl_matrix_set(testing_x[i][0], idx_h, idx_w, atof(ch));
			ch = strtok(NULL, " ");
		}
		free(ch);
	}
	fclose(fp);

	fp = fopen("../rrbm/datasets/mnist_testy.data", "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	gsl_matrix* testing_y = gsl_matrix_calloc(nrow, 10);
	for (int i = 0; i < nrow; i++)
	{
		ssize_t read = getline(&line, &len, fp);
		int y = atoi(line);
		gsl_matrix_set(testing_y, i, y, 1.0);
	}
	fclose(fp);

	printf("Testing Dataset Read\n");

	// Prepare the CNN

	int batch_size = 10;
	int border_mode = 2;
	int filter_size = 5;

	int win_size = 3;
	int stride = 2;

	CONV conv1; create_CONV(&conv1, 1, 4, filter_size, 0.1, border_mode, batch_size);
	POOL pool1; create_POOL(&pool1, 4, 0.1, batch_size, win_size, stride);
	RELU relu1; create_RELU(&relu1, 4, batch_size);
	CONV conv2; create_CONV(&conv2, 4, 16, filter_size, 0.1, border_mode, batch_size);
	POOL pool2; create_POOL(&pool2, 16, 0.1, batch_size, win_size, stride);
	RELU relu2; create_RELU(&relu2, 16, batch_size);
	FLAT flat1; create_FLAT(&flat1, 16, batch_size);
	LINE line1; create_LINE(&line1, 784, 64, 0.01, batch_size);
	RELV relv1; create_RELV(&relv1, batch_size);
	LINE line2; create_LINE(&line2, 64, 10, 0.1, batch_size);
	SOFT soft1; create_SOFT(&soft1, 10, batch_size);

	int num_layers = 11;
	LAYER* layers = (LAYER*) malloc(num_layers * sizeof(LAYER));
	layers[0].type = 1; layers[0].layer = (void*) &conv1;
	layers[1].type = 2; layers[1].layer = (void*) &pool1;
	layers[2].type = 3; layers[2].layer = (void*) &relu1;
	layers[3].type = 1; layers[3].layer = (void*) &conv2;
	layers[4].type = 2; layers[4].layer = (void*) &pool2;
	layers[5].type = 3; layers[5].layer = (void*) &relu2;
	layers[6].type = 4; layers[6].layer = (void*) &flat1;
	layers[7].type = 5; layers[7].layer = (void*) &line1;
	layers[8].type = 8; layers[8].layer = (void*) &relv1;
	layers[9].type = 5; layers[9].layer = (void*) &line2;
	layers[10].type = 6; layers[10].layer = (void*) &soft1;

	printf("CNN created\n");

	// Train a CNN to learn MNIST
	int training_epochs = 15;
	double learning_rate = 1e-2;
	double momentum = 1;

	double loss = train_cnn (training_x, training_y, nrow, num_channels,
		layers, num_layers, training_epochs, batch_size, learning_rate, momentum, 1234);

	printf("CNN trained\n");

	// Free the Network
	free_CONV(&conv1);
	free_POOL(&pool1);
	free_RELU(&relu1);
	free_CONV(&conv2);
	free_POOL(&pool2);
	free_RELU(&relu2);
	free_FLAT(&flat1);
	free_LINE(&line1);
	free_RELV(&relv1);
	free_LINE(&line2);
	free_SOFT(&soft1);

	free(layers);

	// Free the data
	if (line) free(line);

	for (int i = 0; i < 60000; i++)
	{
		gsl_matrix_free(training_x[i][0]);
		free(training_x[i]);
	}
	free(training_x);
	gsl_matrix_free(training_y);

	for (int i = 0; i < 10000; i++)
	{
		gsl_matrix_free(testing_x[i][0]);
		free(testing_x[i]);
	}
	free(testing_x);
	gsl_matrix_free(testing_y);

	return 0;
}

// Driver for Full MLP network using MNIST
int main_mlp()
{
	printf("Start\n");

	int nrow = 60000;
	int ncol = 784;

	int num_channels = 1;
	int img_h = 28;
	int img_w = 28;

	FILE * fp;
	char * line = NULL;
	size_t len = 0;

	// Read Train Data (MNIST)
	fp = fopen("../rrbm/datasets/mnist_trainx.data", "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	gsl_matrix* training_x = gsl_matrix_alloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
	{
		ssize_t read = getline(&line, &len, fp);
		char *ch = strtok(line, " ");
		for (int j = 0; j < ncol; j++)
		{
			gsl_matrix_set(training_x, i, j, atof(ch));
			ch = strtok(NULL, " ");
		}
		free(ch);
	}
	fclose(fp);

	fp = fopen("../rrbm/datasets/mnist_trainy.data", "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	gsl_matrix* training_y = gsl_matrix_calloc(nrow, 10);
	for (int i = 0; i < nrow; i++)
	{
		ssize_t read = getline(&line, &len, fp);
		int y = atoi(line);
		gsl_matrix_set(training_y, i, y, 1.0);
	}
	fclose(fp);

	printf("Training Dataset Read\n");

	// Read Test Data (MNIST)
	fp = fopen("../rrbm/datasets/mnist_testx.data", "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	nrow = 10000;

	gsl_matrix* testing_x = gsl_matrix_alloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
	{
		ssize_t read = getline(&line, &len, fp);
		char *ch = strtok(line, " ");
		for (int j = 0; j < ncol; j++)
		{
			gsl_matrix_set(testing_x, i, j, atof(ch));
			ch = strtok(NULL, " ");
		}
		free(ch);
	}
	fclose(fp);

	fp = fopen("../rrbm/datasets/mnist_testy.data", "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	gsl_matrix* testing_y = gsl_matrix_calloc(nrow, 10);
	for (int i = 0; i < nrow; i++)
	{
		ssize_t read = getline(&line, &len, fp);
		int y = atoi(line);
		gsl_matrix_set(testing_y, i, y, 1.0);
	}
	fclose(fp);

	printf("Testing Dataset Read\n");

	// Prepare the MLP
	int batch_size = 10;

	LINE line1; create_LINE(&line1, 784, 64, 0.01, batch_size);
	RELV relv1; create_RELV(&relv1, batch_size);
	LINE line2; create_LINE(&line2, 64, 10, 0.1, batch_size);
	SOFT soft1; create_SOFT(&soft1, 10, batch_size);

	LAYER* layers = (LAYER*) malloc(4 * sizeof(LAYER));
	layers[0].type = 5; layers[0].layer = (void*) &line1;
	layers[1].type = 8; layers[1].layer = (void*) &relv1;
	layers[2].type = 5; layers[2].layer = (void*) &line2;
	layers[3].type = 6; layers[3].layer = (void*) &soft1;

	printf("MLP created\n");

	// Train a MLP to learn MNIST
	int training_epochs = 20;
	double learning_rate = 1e-2;
	double momentum = 1;

	double loss = train_mlp (training_x, training_y, layers, 4,
		training_epochs, batch_size, learning_rate, momentum, 1234);

	printf("MLP trained\n");

	// Free the Network
	free_LINE(&line1);
	free_RELV(&relv1);
	free_LINE(&line2);
	free_SOFT(&soft1);

	free(layers);

	// Free the data
	if (line) free(line);

	gsl_matrix_free(training_x);
	gsl_matrix_free(training_y);

	gsl_matrix_free(testing_x);
	gsl_matrix_free(testing_y);

	return 0;
}

/*---------------------------------------------------------------------------*/
/* MAIN FUNCTION - TEST                                                      */
/*---------------------------------------------------------------------------*/

int main()
{
	return main_cnn();
}
