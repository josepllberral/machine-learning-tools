/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Function Implementations
// Compile using "gcc cnn.c conv.c pool.c flat.c relu.c relv.c line.c soft.c cell.c grad_check.c matrix_ops.c -lgsl -lgslcblas -lm -o cnn"

#include "cnn.h"

/*---------------------------------------------------------------------------*/
/* AUXILIAR FUNCTIONS                                                        */
/*---------------------------------------------------------------------------*/

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
/* PIPELINE LAYER HANDLER                                                    */
/*---------------------------------------------------------------------------*/

void* forward (LAYER* layer, void* batchdata)
{
	void* retval;

	int size1 = 0;
	int size2 = 0;

	switch (layer->type)
	{
		case 1:	;
			CONV* conv = (CONV*) layer;
			gsl_matrix*** x1 = (gsl_matrix***) batchdata;
			retval = (void*) forward_conv(conv, x1);
			size1 = conv->batch_size;
			size2 = conv->n_channels;
			break;
		case 2:	;
			POOL* pool = (POOL*) layer;
			gsl_matrix*** x2 = (gsl_matrix***) batchdata;
			retval = (void*) forward_pool(pool, x2);
			size1 = pool->batch_size;
			size2 = pool->n_channels;
			break;
		case 3:	;
			RELU* relu = (RELU*) layer;
			gsl_matrix*** x3 = (gsl_matrix***) batchdata;
			retval = (void*) forward_relu(relu, x3);
			size1 = relu->batch_size;
			size2 = relu->n_channels;
			break;
		case 4: ;
			FLAT* flat = (FLAT*) layer;
			gsl_matrix*** x4 = (gsl_matrix***) batchdata;
			retval = (void*) forward_flat(flat, x4);
			size1 = flat->batch_size;
			size2 = flat->n_channels;
			break;
		case 5: ;
			LINE* line = (LINE*) layer;
			gsl_matrix* x5 = (gsl_matrix*) batchdata;
			retval = (void*) forward_line(line, x5);
			break;
		case 6: ;
			SOFT* soft = (SOFT*) layer;
			gsl_matrix* x6 = (gsl_matrix*) batchdata;
			retval = (void*) forward_soft(soft, x6);
			break;
		default:
			retval = NULL;
	}

	switch (layer->type)
	{
		case 1:
		case 2:
		case 3:
		case 4: ;
			gsl_matrix*** aux = (gsl_matrix***) batchdata;
			for (int i = 0; i < size1; i++)
			{
				for (int j = 0; j < size2; j++) gsl_matrix_free(aux[i][j]);
				free(aux[i]);
			}
			free(aux);
			break;
		case 5:
		case 6:
			gsl_matrix_free((gsl_matrix*) batchdata);
			break;
		default:
			break;
	}

	return retval;
}

void* backward (LAYER* layer, void* negdata)
{
	void* retval;

	int size1 = 0;
	int size2 = 0;

	switch (layer->type)
	{
		case 1: ;
			CONV* conv = (CONV*) layer;
			gsl_matrix*** x1 = (gsl_matrix***) negdata;
			retval = (void*) backward_conv(conv, x1);
			size1 = conv->batch_size;
			size2 = conv->n_channels;
			break;
		case 2: ;
			POOL* pool = (POOL*) layer;
			gsl_matrix*** x2 = (gsl_matrix***) negdata;
			retval = (void*) backward_pool(pool, x2);
			size1 = pool->batch_size;
			size2 = pool->n_channels;
			break;
		case 3: ;
			RELU* relu = (RELU*) layer;
			gsl_matrix*** x3 = (gsl_matrix***) negdata;
			retval = (void*) backward_relu(relu, x3);
			size1 = relu->batch_size;
			size2 = relu->n_channels;
			break;
		case 4: ;
			FLAT* flat = (FLAT*) layer;
			gsl_matrix* x4 = (gsl_matrix*) negdata;
			retval = (void*) backward_flat(flat, x4);
			break;
		case 5: ;
			LINE* line = (LINE*) layer;
			gsl_matrix* x5 = (gsl_matrix*) negdata;
			retval = (void*) backward_line(line, x5);
			break;
		case 6: ;
			SOFT* soft = (SOFT*) layer;
			gsl_matrix* x6 = (gsl_matrix*) negdata;
			retval = (void*) backward_soft(soft, x6);
			break;
		default:
			retval = NULL;
	}

	switch (layer->type)
	{
		case 1:
		case 2:
		case 3: ;
			gsl_matrix*** aux = (gsl_matrix***) negdata;
			for (int i = 0; i < size1; i++)
			{
				for (int j = 0; j < size2; j++) gsl_matrix_free(aux[i][j]);
				free(aux[i]);
			}
			free(aux);
			break;
		case 4:
		case 5:
		case 6:
			gsl_matrix_free((gsl_matrix*) negdata);
			break;
		default:
			break;
	}

	return retval;
}

void get_updates (LAYER* layer, double learning_rate)
{
	switch (layer->type)
	{
		case 1:
			get_updates_conv((CONV*) layer, learning_rate);
			break;
		case 2:
			get_updates_pool((POOL*) layer, learning_rate);
			break;
		case 3:
			get_updates_relu((RELU*) layer, learning_rate);
			break;
		case 4:
			get_updates_flat((FLAT*) layer, learning_rate);
			break;
		case 5:
			get_updates_line((LINE*) layer, learning_rate);
			break;
		case 6:
			get_updates_soft((SOFT*) layer, learning_rate);
			break;
		default:
			break;
	}
}

/*---------------------------------------------------------------------------*/
/* HOW TO TRAIN YOUR CNN                                                     */
/*---------------------------------------------------------------------------*/

// Function to train the CNN
//  param training_x      : loaded dataset (rows = examples, cols = features)
//  param training_y      : loaded labels (binarized vector into rows = examples, cols = labels)
//  param num_samples     : number of image samples
//  param num_channels    : number of channels per image
//  param layers          : array of layers
//  param num_layers      : number of layers in array
//  param training_epochs : number of epochs used for training
//  param batch_size      : size of a batch used to train the CNN
//  param learning_rate   : learning rate used for training the CNN
//  param momentum        : momentum rate used for training the CNN (Currently not used)
//  param rand_seed       : random seed for training
//  returns               : average loss in minibatches
double train_cnn (gsl_matrix*** training_x, gsl_matrix* training_y, int num_samples,
	int num_channels, LAYER* layers, int num_layers, int training_epochs,
	int batch_size, double learning_rate, double momentum, int rand_seed)
{
	srand(rand_seed);

	int num_batches = num_samples / batch_size;

	int img_h = training_x[0][0]->size1;
	int img_w = training_x[0][0]->size2;

	int out_size = training_y->size2;

	CELL loss_layer;
	create_CELL(&loss_layer);

	double acc_loss = 0;

	for (int epoch = 0; epoch < training_epochs; epoch++)
	{
//		#confusion = ConfusionMatrix(num_classes)

		acc_loss = 0;
		for (int j = 0; j < num_batches; j++)
		{
			// Select mini_batch
			int idx_ini = j * batch_size;
			int idx_fin = idx_ini + batch_size - 1;

			if (idx_fin >= num_samples) break;

			gsl_matrix*** minibatch = (gsl_matrix***) malloc(batch_size * sizeof(gsl_matrix**));
			for (int b = 0; b < batch_size; b++)
			{
				minibatch[b] = (gsl_matrix**) malloc(num_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < num_channels; c++)
				{
					minibatch[b][c] = gsl_matrix_alloc(img_h, img_w);
					gsl_matrix_memcpy(minibatch[b][c], training_x[idx_ini + b][c]);
				}
			}

			gsl_matrix* targets = gsl_matrix_alloc(batch_size, out_size);
			for (int b = 0; b < batch_size; b++)
			{
				gsl_vector* aux = gsl_vector_alloc(out_size);
				gsl_matrix_get_row(aux, training_y, b);
				gsl_matrix_set_row(targets, b, aux);
				gsl_vector_free(aux);
			}

			// Forward through layers
			void* batchdata = (void*) minibatch;
			for (int i = 0; i < num_layers; i++)
				batchdata = forward(&layers[i], batchdata);

			// Calculate Forward Loss
			gsl_matrix* output = (gsl_matrix*) batchdata;
			gsl_matrix* y = forward_cell(&loss_layer, output, targets);
			gsl_matrix_free(y);

			// Calculate Negdata
			void* negdata = (void*) backward_cell(&loss_layer, output, targets);
			gsl_matrix_free(output);
			gsl_matrix_free(targets);

			// Backward through layers
			for (int i = num_layers - 1; i >= 0; i--)
				negdata = backward(&layers[i], negdata);

			// Update layers
			for (int i = 0; i < num_layers; i++)
				get_updates(&layers[i], learning_rate);

//			#confusion.batch_add(target_batch.argmax(-1), y_probs.argmax(-1))
			acc_loss += loss_layer.loss;
		}

		if (epoch % 1 == 0)
		{
//			#curr_acc = confusion.accuracy() # TODO
			printf("Epoch %d: Mean Loss %f", epoch, acc_loss / num_batches);
		}
	}

	free_CELL(&loss_layer);

	return (acc_loss / num_batches);
}

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
		training_x[i] = (gsl_matrix**) malloc(1 * sizeof(gsl_matrix*)); // num_channels = 1

		ssize_t read = getline(&line, &len, fp);
		char *ch = strtok(line, " ");
		for (int j = 0; j < ncol; j++)
		{
			int idx_h = j % img_h;
			int idx_w = j / img_w;
			training_x[i][0] = gsl_matrix_alloc(img_h, img_w);
			gsl_matrix_set(training_x[i][0], idx_h, idx_w, atof(ch));
			ch = strtok(NULL, " ");
		}
		free(ch);
	}
	fclose(fp);
/*
	fp = fopen("../rrbm/datasets/mnist_trainy.data", "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	gsl_matrix* training_y = gsl_matrix_calloc(nrow, 10);
	for (int i = 0; i < nrow; i++)
	{
		ssize_t read = getline(&line, &len, fp);
		char *ch = strtok(line, " ");
		int y = atoi(ch);
		gsl_matrix_set(training_y, i, y, 1.0);
		free(ch);
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
		testing_x[i] = (gsl_matrix**) malloc(1 * sizeof(gsl_matrix*)); // num_channels = 1

		ssize_t read = getline(&line, &len, fp);
		char *ch = strtok(line, " ");
		for (int j = 0; j < ncol; j++)
		{
			int idx_h = j % img_h;
			int idx_w = j / img_w;
			testing_x[i][0] = gsl_matrix_alloc(img_h, img_w);
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
		char *ch = strtok(line, " ");
		int y = atoi(ch);
		gsl_matrix_set(testing_y, i, y, 1.0);
		free(ch);
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
	LINE line1; create_LINE(&line1, 748, 64, 0.01, batch_size);
	RELV relv1; create_RELV(&relv1);
	LINE line2; create_LINE(&line1, 64, 10, 0.1, batch_size);
	SOFT soft1; create_SOFT(&soft1, 10, batch_size);

	LAYER* layers = (LAYER*) malloc(11 * sizeof(LAYER));
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
	int training_epochs = 10;
	double learning_rate = 5e-3;
	double momentum = 1;

	double loss = train_cnn (training_x, training_y, nrow, num_channels,
		layers, 11, training_epochs, batch_size, learning_rate, momentum, 1234);


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
*/
	// Free the data
	for (int i = 0; i < 60000; i++)
	{
		gsl_matrix_free(training_x[i][0]);
		free(training_x[i]);
	}
	free(training_x);
/*	gsl_matrix_free(training_y);

	for (int i = 0; i < 10000; i++)
	{
		gsl_matrix_free(testing_x[i][0]);
		free(testing_x[i]);
	}
	free(testing_x);
	gsl_matrix_free(testing_y);
*/
}

int main()
{
	return main_cnn();
}
