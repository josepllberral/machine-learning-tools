/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Convolutional Neural Network Function Implementations

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

// Function to compute the Classification Accuracy
// param predicted : matrix with results. The "predicted result" will be the MAX
// param observed  : matrix with real values. Supposing that each row is one-hot-encoded
//                   the real value is also the MAX, with expected value 1.
double classification_accuracy (gsl_matrix* predicted, gsl_matrix* observed)
{
	int nrows = predicted->size1;
	int ncols = predicted->size2;

	int correct = 0;

	for (int i = 0; i < nrows; i++)
	{
		gsl_vector* prv = gsl_vector_alloc(ncols);
		gsl_vector* obv = gsl_vector_alloc(ncols);

		gsl_matrix_get_row(prv, predicted, i);
		gsl_matrix_get_row(obv, observed, i);

		int idx_maxobs = gsl_vector_max_index(prv);
		int idx_maxpred = gsl_vector_max_index(obv);

		if (idx_maxobs == idx_maxpred) correct++;

		gsl_vector_free(prv);
		gsl_vector_free(obv);
	}

	return ((double) correct / (double) nrows);
}

// Function to print the Confusion Matrix
void classification_matrix_print (gsl_matrix* predicted, gsl_matrix* observed)
{
	int nrows = predicted->size1;
	int ncols = predicted->size2;

	gsl_matrix* confusion = gsl_matrix_calloc(ncols, ncols);

	for (int i = 0; i < nrows; i++)
	{
		gsl_vector* prv = gsl_vector_alloc(ncols);
		gsl_vector* obv = gsl_vector_alloc(ncols);

		gsl_matrix_get_row(prv, predicted, i);
		gsl_matrix_get_row(obv, observed, i);

		int idx_maxobs = gsl_vector_max_index(obv);
		int idx_maxpred = gsl_vector_max_index(prv);

		double value = gsl_matrix_get(confusion, idx_maxobs, idx_maxpred);
		gsl_matrix_set(confusion, idx_maxobs, idx_maxpred, value + 1);

		gsl_vector_free(prv);
		gsl_vector_free(obv);
	}

	printf("Observed VVV \\ Predicted >>>\n");
	for (int i = 0; i < ncols; i++)
	{
		for (int j = 0; j < ncols; j++)
			printf("%d ", (int)floor(gsl_matrix_get(confusion, i, j)));
		printf("\n");
	}
	printf("--------------------------------------\n");
}

// Function to print a GSL Matrix
void print_matrix (gsl_matrix* x)
{
	for (int i = 0; i < x->size1; i++)
	{
		for (int j = 0; j < x->size2; j++)
			printf("%f ", gsl_matrix_get(x, i, j));
		printf("\n");
	}
	printf("-------------\n");
}

// Function to print a frame in a 4D image
void print_image00 (gsl_matrix*** x, int a, int b)
{
	for (int i = 0; i < x[a][b]->size1; i++)
	{
		for (int j = 0; j < x[a][b]->size2; j++)
			printf("%f ", gsl_matrix_get(x[a][b], i, j));
		printf("\n");
	}
	printf("-------------\n");
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
		case 9: ;
			SIGM* sigm = (SIGM*) layer->layer;
			gsl_matrix* x9 = batchdata->matrix;
			gsl_matrix* y9 = forward_sigm(sigm, x9);
			gsl_matrix_free(batchdata->matrix);
			batchdata->matrix = y9;
			break;
		case 11: ;
			DIRE* dire = (DIRE*) layer->layer;
			gsl_matrix* x11 = batchdata->matrix;
			gsl_matrix* y11 = forward_dire(dire, x11);
			gsl_matrix_free(batchdata->matrix);
			batchdata->matrix = y11;
			break;
		default:
			break;
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
		case 9: ;
			SIGM* sigm = (SIGM*) layer->layer;
			gsl_matrix* y9 = negdata->matrix;
			gsl_matrix* x9 = backward_sigm(sigm, y9);
			gsl_matrix_free(negdata->matrix);
			negdata->matrix = x9;
			break;
		case 11: ;
			DIRE* dire = (DIRE*) layer->layer;
			gsl_matrix* y11 = negdata->matrix;
			gsl_matrix* x11 = backward_dire(dire, y11);
			gsl_matrix_free(negdata->matrix);
			negdata->matrix = x11;
			break;
		default:
			break;
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
		case 9:
			get_updates_sigm((SIGM*) layer->layer, learning_rate);
			break;
		case 11:
			get_updates_dire((DIRE*) layer->layer, learning_rate);
			break;
		default:
			break;
	}
}

void update_batch_size (LAYER* layer, int batch_size)
{
	switch (layer->type)
	{
		case 1: ;
			CONV* aux1 = (CONV*) layer->layer;

			for (int b = 0; b < aux1->batch_size; b++)
			{
				for (int c = 0; c < aux1->n_channels; c++) gsl_matrix_free(aux1->img[b][c]);
				free(aux1->img[b]);
			}
			free(aux1->img);

			aux1->batch_size = batch_size;

			aux1->img = (gsl_matrix***) malloc(aux1->batch_size * sizeof(gsl_matrix**));
			for (int b = 0; b < aux1->batch_size; b++)
			{
				aux1->img[b] = (gsl_matrix**) malloc(aux1->n_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < aux1->n_channels; c++)
					aux1->img[b][c] = gsl_matrix_calloc(1, 1);
			}

			break;
		case 2: ;
			POOL* aux2 = (POOL*) layer->layer;

			for (int b = 0; b < aux2->batch_size; b++)
			{
				for (int c = 0; c < aux2->n_channels; c++) gsl_matrix_free(aux2->img[b][c]);
				free(aux2->img[b]);
			}
			free(aux2->img);

			aux2->batch_size = batch_size;

			aux2->img = (gsl_matrix***) malloc(aux2->batch_size * sizeof(gsl_matrix**));
			for (int b = 0; b < aux2->batch_size; b++)
			{
				aux2->img[b] = (gsl_matrix**) malloc(aux2->n_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < aux2->n_channels; c++)
					aux2->img[b][c] = gsl_matrix_calloc(1, 1);
			}

			break;
		case 3: ;
			RELU* aux3 = (RELU*) layer->layer;

			for (int b = 0; b < aux3->batch_size; b++)
			{
				for (int c = 0; c < aux3->n_channels; c++) gsl_matrix_free(aux3->img[b][c]);
				free(aux3->img[b]);
			}
			free(aux3->img);

			aux3->batch_size = batch_size;

			aux3->img = (gsl_matrix***) malloc(aux3->batch_size * sizeof(gsl_matrix**));
			for (int b = 0; b < aux3->batch_size; b++)
			{
				aux3->img[b] = (gsl_matrix**) malloc(aux3->n_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < aux3->n_channels; c++)
					aux3->img[b][c] = gsl_matrix_calloc(1, 1);
			}

			break;
		case 4: ;
			FLAT* aux4 = (FLAT*) layer->layer;
			aux4->batch_size = batch_size;
			break;
		case 5: ;
			LINE* aux5 = (LINE*) layer->layer;

			gsl_matrix_free(aux5->x);

			aux5->batch_size = batch_size;

			aux5->x = gsl_matrix_calloc(aux5->batch_size, aux5->n_visible);

			break;
		case 6: ;
			SOFT* aux6 = (SOFT*) layer->layer;
			aux6->batch_size = batch_size;
			break;
		case 7: ;
			break;
		case 8: ;
			RELV* aux8 = (RELV*) layer->layer;
			aux8->batch_size = batch_size;
			break;
		case 9: ;
			SIGM* aux9 = (SIGM*) layer->layer;

			gsl_matrix_free(aux9->a);

			aux9->batch_size = batch_size;

			aux9->a = gsl_matrix_calloc(aux9->batch_size, aux9->n_units);

			break;
		case 11: ;
			DIRE* aux11 = (DIRE*) layer->layer;

			gsl_matrix_free(aux11->buff_x);
			gsl_matrix_free(aux11->buff_dy);

			aux11->batch_size = batch_size;

			aux11->buff_x = gsl_matrix_calloc(aux11->batch_size, aux11->n_units);
			aux11->buff_dy = gsl_matrix_calloc(aux11->batch_size, aux11->n_units);

			break;
		default: ;
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
	double acc_class = 0;

	for (int epoch = 0; epoch < training_epochs; epoch++)
	{
		acc_loss = 0;
		acc_class = 0;

		for (int j = 0; j < num_batches; j++)
		{
			// Select mini_batch
			int idx_ini = j * batch_size;
			int idx_fin = idx_ini + batch_size - 1;

			if (idx_fin >= num_samples) break;

			gsl_matrix*** minibatch = (gsl_matrix***) malloc(batch_size * sizeof(gsl_matrix**));
			gsl_matrix* targets = gsl_matrix_alloc(batch_size, out_size);
			for (int b = 0; b < batch_size; b++)
			{
				minibatch[b] = (gsl_matrix**) malloc(num_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < num_channels; c++)
				{
					minibatch[b][c] = gsl_matrix_alloc(img_h, img_w);
					gsl_matrix_memcpy(minibatch[b][c], training_x[idx_ini + b][c]);
				}
				gsl_vector* aux = gsl_vector_alloc(out_size);
				gsl_matrix_get_row(aux, training_y, idx_ini + b);
				gsl_matrix_set_row(targets, b, aux);
				gsl_vector_free(aux);
			}

			// Forward through layers
			batchdata.image = minibatch;
			for (int i = 0; i < num_layers; i++)
				forward(&(layers[i]), &batchdata);

			// Calculate Forward Loss and Negdata
			gsl_matrix* output = batchdata.matrix;
			gsl_matrix* pred_y = forward_cell(&loss_layer, output, targets);
			gsl_matrix* results = backward_cell(&loss_layer, output, targets);

			acc_loss += loss_layer.loss;
			acc_class += classification_accuracy(pred_y, targets);

//			if (j == num_batches - 1 && epoch == training_epochs - 1)
//			{
//				printf("Last batch confusion matrix:");
//				classification_matrix_print(pred_y, targets);
//			}

			gsl_matrix_free(pred_y);
			gsl_matrix_free(output);
			gsl_matrix_free(targets);

			// Backward through layers, and update them
			batchdata.matrix = results;
			for (int i = num_layers - 1; i >= 0; i--)
			{
				backward(&(layers[i]), &batchdata);
				get_updates(&(layers[i]), learning_rate);
			}
		}

//		if (epoch % 1 == 0)
			printf("Epoch %d: Mean Loss %f, Classification Accuracy %f\n", epoch, acc_loss / num_batches, acc_class / num_batches);
	}

	free_CELL(&loss_layer);

	return (acc_loss / num_batches);
}

/*---------------------------------------------------------------------------*/
/* PREDICTION USING THE CNN                                                  */
/*---------------------------------------------------------------------------*/

// Function to predict the results of a matrix
gsl_matrix* prediction_cnn (gsl_matrix*** testing_x, int num_samples,
	int num_channels, LAYER* layers, int num_layers)
{
	int batch_size = min(100, num_samples);
	int num_batches = num_samples / batch_size;
	if (num_samples % batch_size > 0) num_batches++;

	int num_outputs = 1;
	if (layers[num_layers - 1].type == 5) // LINE
		num_outputs = ((LINE*)(layers[num_layers - 1].layer))->n_hidden;
	else if (layers[num_layers - 1].type == 6) // SOFT
		num_outputs = ((SOFT*)(layers[num_layers - 1].layer))->n_units;
	else if (layers[num_layers - 1].type == 9) // SIGM
		num_outputs = ((SIGM*)(layers[num_layers - 1].layer))->n_units;
	else if (layers[num_layers - 1].type == 11) // DIRE
		num_outputs = ((DIRE*)(layers[num_layers - 1].layer))->n_units;

	gsl_matrix* result = gsl_matrix_alloc(num_samples, num_outputs);

	int img_h = testing_x[0][0]->size1;
	int img_w = testing_x[0][0]->size2;

	data batchdata;

	// Update batch_size for layers
	for (int i = 0; i < num_layers; i++)
		update_batch_size(&(layers[i]), batch_size);

	// Loop through examples
	for (int j = 0; j < num_batches; j++)
	{
		// Select mini_batch
		int idx_ini = j * batch_size;
		int idx_fin = idx_ini + batch_size - 1;

		int real_batch_size = batch_size;
		if (idx_fin >= num_samples) real_batch_size = num_samples % batch_size;

		gsl_matrix*** minibatch = (gsl_matrix***) malloc(batch_size * sizeof(gsl_matrix**));
		for (int b = 0; b < real_batch_size; b++)
		{
			minibatch[b] = (gsl_matrix**) malloc(num_channels * sizeof(gsl_matrix*));
			for (int c = 0; c < num_channels; c++)
			{
				minibatch[b][c] = gsl_matrix_alloc(img_h, img_w);
				gsl_matrix_memcpy(minibatch[b][c], testing_x[idx_ini + b][c]);
			}
		}

		// Completar el Mini-Batch
		if (batch_size > real_batch_size)
			for (int b = real_batch_size ; b < batch_size; b++)
			{
				minibatch[b] = (gsl_matrix**) malloc(num_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < num_channels; c++)
					minibatch[b][c] = gsl_matrix_calloc(img_h, img_w);
			}

		// Forward through layers
		batchdata.image = minibatch;
		for (int i = 0; i < num_layers; i++)
			forward(&(layers[i]), &batchdata);

		// Calculate Forward Loss and Negdata
		gsl_matrix* output = batchdata.matrix;

		// Add output to results
		for (int b = 0; b < real_batch_size; b++)
		{
			gsl_vector* aux = gsl_vector_alloc(num_outputs);
			gsl_matrix_get_row(aux, output, b);
			gsl_matrix_set_row(result, j * batch_size + b, aux);
			gsl_vector_free(aux);
		}

		gsl_matrix_free(output);
	}

	return result;
}

