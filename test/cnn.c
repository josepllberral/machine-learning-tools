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
	if ((*destination))
	{
		for (int b = 0; b < size1; b++)
			if ((*destination)[b])
			{
				for (int c = 0; c < size2; c++)
					if ((*destination)[b][c]) gsl_matrix_free((*destination)[b][c]);
				free((*destination)[b]);
			}
		free((*destination));
	}
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

void forward (LAYER* layer, data* batchdata)
{
	switch (layer->type)
	{
		case 1:	;
			CONV* conv = (CONV*) layer->layer;
			gsl_matrix*** x1 = batchdata->image;
			gsl_matrix*** y1 = forward_conv(conv, x1);
			for (int i = 0; i < conv->batch_size; i++)
			{
				for (int j = 0; j < conv->n_channels; j++) gsl_matrix_free(batchdata->image[i][j]);
				free(batchdata->image[i]);
			}
			free(batchdata->image);
			batchdata->image = y1;
			break;
		case 2:	;
			POOL* pool = (POOL*) layer->layer;
			gsl_matrix*** x2 = batchdata->image;
			gsl_matrix*** y2 = forward_pool(pool, x2);
			for (int i = 0; i < pool->batch_size; i++)
			{
				for (int j = 0; j < pool->n_channels; j++) gsl_matrix_free(batchdata->image[i][j]);
				free(batchdata->image[i]);
			}
			free(batchdata->image);
			batchdata->image = y2;
			break;
		case 3:	;
			RELU* relu = (RELU*) layer->layer;
			gsl_matrix*** x3 = batchdata->image;
			gsl_matrix*** y3 = forward_relu(relu, x3);
			for (int i = 0; i < relu->batch_size; i++)
			{
				for (int j = 0; j < relu->n_channels; j++) gsl_matrix_free(batchdata->image[i][j]);
				free(batchdata->image[i]);
			}
			free(batchdata->image);
			batchdata->image = y3;
			break;
		case 4: ;
			FLAT* flat = (FLAT*) layer->layer;
			gsl_matrix*** x4 = batchdata->image;
			gsl_matrix* y4 = forward_flat(flat, x4);
			for (int i = 0; i < flat->batch_size; i++)
			{
				for (int j = 0; j < flat->n_channels; j++) gsl_matrix_free(batchdata->image[i][j]);
				free(batchdata->image[i]);
			}
			free(batchdata->image);
			batchdata->matrix = y4;
			break;
		case 5: ;
			LINE* line = (LINE*) layer->layer;
			gsl_matrix* x5 = batchdata->matrix;
			gsl_matrix* y5 = forward_line(line, x5);
			gsl_matrix_free(batchdata->matrix);
			batchdata->matrix = y5;
			break;
		case 6: ;
			SOFT* soft = (SOFT*) layer->layer;
			gsl_matrix* x6 = batchdata->matrix;
			gsl_matrix* y6 = forward_soft(soft, x6);
			gsl_matrix_free(batchdata->matrix);
			batchdata->matrix = y6;
			break;
		case 7: ;
			break;
		case 8: ;
			RELV* relv = (RELV*) layer->layer;
			gsl_matrix* x8 = batchdata->matrix;
			gsl_matrix* y8 = forward_relv(relv, x8);
			gsl_matrix_free(batchdata->matrix);
			batchdata->matrix = y8;
			break;
		default:
			batchdata->matrix = NULL;
	}
}

void backward (LAYER* layer, data* negdata)
{
	switch (layer->type)
	{
		case 1: ;
			CONV* conv = (CONV*) layer->layer;
			gsl_matrix*** y1 = negdata->image;
			gsl_matrix*** x1 = backward_conv(conv, y1);
			for (int i = 0; i < conv->batch_size; i++)
			{
				for (int j = 0; j < conv->n_channels; j++) gsl_matrix_free(negdata->image[i][j]);
				free(negdata->image[i]);
			}
			free(negdata->image);
			negdata->image = x1;
			break;
		case 2: ;
			POOL* pool = (POOL*) layer->layer;
			gsl_matrix*** y2 = negdata->image;
			gsl_matrix*** x2 = backward_pool(pool, y2);
			for (int i = 0; i < pool->batch_size; i++)
			{
				for (int j = 0; j < pool->n_channels; j++) gsl_matrix_free(negdata->image[i][j]);
				free(negdata->image[i]);
			}
			free(negdata->image);
			negdata->image = x2;
			break;
		case 3: ;
			RELU* relu = (RELU*) layer->layer;
			gsl_matrix*** y3 = negdata->image;
			gsl_matrix*** x3 = backward_relu(relu, y3);
			for (int i = 0; i < relu->batch_size; i++)
			{
				for (int j = 0; j < relu->n_channels; j++) gsl_matrix_free(negdata->image[i][j]);
				free(negdata->image[i]);
			}
			free(negdata->image);
			negdata->image = x3;
			break;
		case 4: ;
			FLAT* flat = (FLAT*) layer->layer;
			gsl_matrix* y4 = negdata->matrix;
			gsl_matrix*** x4 = backward_flat(flat, y4);
			gsl_matrix_free(negdata->matrix);
			negdata->image = x4;
			break;
		case 5: ;
			LINE* line = (LINE*) layer->layer;
			gsl_matrix* y5 = negdata->matrix;
			gsl_matrix* x5 = backward_line(line, y5);
			gsl_matrix_free(negdata->matrix);
			negdata->matrix = x5;
			break;
		case 6: ;
			SOFT* soft = (SOFT*) layer->layer;
			gsl_matrix* y6 = negdata->matrix;
			gsl_matrix* x6 = backward_soft(soft, y6);
			gsl_matrix_free(negdata->matrix);
			negdata->matrix = x6;
			break;
		case 7: ;
			break;
		case 8: ;
			RELV* relv = (RELV*) layer->layer;
			gsl_matrix* y8 = negdata->matrix;
			gsl_matrix* x8 = backward_relv(relv, y8);
			gsl_matrix_free(negdata->matrix);
			negdata->matrix = x8;
			break;
		default:
			negdata->matrix = NULL;
	}
}

void get_updates (LAYER* layer, double learning_rate)
{
	switch (layer->type)
	{
		case 1:
			get_updates_conv((CONV*) layer->layer, learning_rate);
			break;
		case 2:
			get_updates_pool((POOL*) layer->layer, learning_rate);
			break;
		case 3:
			get_updates_relu((RELU*) layer->layer, learning_rate);
			break;
		case 4:
			get_updates_flat((FLAT*) layer->layer, learning_rate);
			break;
		case 5:
			get_updates_line((LINE*) layer->layer, learning_rate);
			break;
		case 6:
			get_updates_soft((SOFT*) layer->layer, learning_rate);
			break;
		case 7:
			break;
		case 8:
			get_updates_relv((RELV*) layer->layer, learning_rate);
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

	data batchdata;

	double acc_loss = 0;

	for (int epoch = 0; epoch < training_epochs; epoch++)
	{
		acc_loss = 0;
		for (int j = 0; j < num_batches; j++)
		{
			if (j % 20 == 0) printf("Batch number %d\n", j);

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
			batchdata.image = minibatch;
			for (int i = 0; i < num_layers; i++)
				forward(&(layers[i]), &batchdata);

			// Calculate Forward Loss and Negdata
			gsl_matrix* output = batchdata.matrix;
			gsl_matrix* y = forward_cell(&loss_layer, output, targets);
			gsl_matrix* results = backward_cell(&loss_layer, output, targets);

			gsl_matrix_free(y);
			gsl_matrix_free(output);
			gsl_matrix_free(targets);

			// Backward through layers
			batchdata.matrix = results;
			for (int i = num_layers - 1; i >= 0; i--)
				backward(&(layers[i]), &batchdata);

			// Update layers
			for (int i = 0; i < num_layers; i++)
				get_updates(&(layers[i]), learning_rate);

			acc_loss += loss_layer.loss;
		}

		if (epoch % 1 == 0)
		{
			// TODO - Compute Prediction Accuracy
			printf("Epoch %d: Mean Loss %f", epoch, acc_loss / num_batches);
		}
	}

	free_CELL(&loss_layer);

	return (acc_loss / num_batches);
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

	int batch_size = 100;
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

int main()
{
	return main_cnn();
}
