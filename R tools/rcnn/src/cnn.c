/*----------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                          */
/*----------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including Convolutional Neural Network Function Implementations

#include "cnn.h"

/*----------------------------------------------------------------------------*/
/* AUXILIAR FUNCTIONS                                                         */
/*----------------------------------------------------------------------------*/

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

// Function to produce a Confusion Matrix
gsl_matrix* classification_matrix (gsl_matrix* predicted, gsl_matrix* observed)
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

	return confusion; // Oh my...!
}

// Function to print the Confusion Matrix
void classification_matrix_print (gsl_matrix* predicted, gsl_matrix* observed)
{
	int ncols = predicted->size2;

	gsl_matrix* confusion = classification_matrix(predicted, observed);

	printf("Observed VVV \\ Predicted >>>\n");
	for (int i = 0; i < ncols; i++)
	{
		for (int j = 0; j < ncols; j++)
			printf("%d ", (int)floor(gsl_matrix_get(confusion, i, j)));
		printf("\n");
	}
	printf("--------------------------------------\n");

	gsl_matrix_free(confusion);
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

/*----------------------------------------------------------------------------*/
/* PIPELINE LAYER HANDLER                                                     */
/*----------------------------------------------------------------------------*/

void forward (LAYER* layer, data* batchdata, int* batch_chan)
{
	switch (layer->type)
	{
		case 1:	;
			CONV* conv = (CONV*) layer->layer;
			gsl_matrix*** x1 = batchdata->image;
			gsl_matrix*** y1 = forward_conv(conv, x1);
			for (int i = 0; i < conv->batch_size; i++)
			{
				for (int j = 0; j < *batch_chan; j++) gsl_matrix_free(batchdata->image[i][j]);
				free(batchdata->image[i]);
			}
			free(batchdata->image);
			batchdata->image = y1;
			*batch_chan = conv->n_filters;
			break;
		case 2:	;
			POOL* pool = (POOL*) layer->layer;
			gsl_matrix*** x2 = batchdata->image;
			gsl_matrix*** y2 = forward_pool(pool, x2);
			for (int i = 0; i < pool->batch_size; i++)
			{
				for (int j = 0; j < *batch_chan; j++) gsl_matrix_free(batchdata->image[i][j]);
				free(batchdata->image[i]);
			}
			free(batchdata->image);
			batchdata->image = y2;
			*batch_chan = pool->n_channels;
			break;
		case 3:	;
			RELU* relu = (RELU*) layer->layer;
			gsl_matrix*** x3 = batchdata->image;
			gsl_matrix*** y3 = forward_relu(relu, x3);
			for (int i = 0; i < relu->batch_size; i++)
			{
				for (int j = 0; j < *batch_chan; j++) gsl_matrix_free(batchdata->image[i][j]);
				free(batchdata->image[i]);
			}
			free(batchdata->image);
			batchdata->image = y3;
			*batch_chan = relu->n_channels;
			break;
		case 4: ;
			FLAT* flat = (FLAT*) layer->layer;
			gsl_matrix*** x4 = batchdata->image;
			gsl_matrix* y4 = forward_flat(flat, x4);
			for (int i = 0; i < flat->batch_size; i++)
			{
				for (int j = 0; j < *batch_chan; j++) gsl_matrix_free(batchdata->image[i][j]);
				free(batchdata->image[i]);
			}
			free(batchdata->image);
			batchdata->matrix = y4;
			*batch_chan = 0;
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
		case 12: ;
			TANH* tanh = (TANH*) layer->layer;
			gsl_matrix* x12 = batchdata->matrix;
			gsl_matrix* y12 = forward_tanh(tanh, x12);
			gsl_matrix_free(batchdata->matrix);
			batchdata->matrix = y12;
			break;
		case 13: ;
			RBML* rbml = (RBML*) layer->layer;
			gsl_matrix* x13 = batchdata->matrix;
			gsl_matrix* y13 = forward_rbml(rbml, x13);
			gsl_matrix_free(batchdata->matrix);
			batchdata->matrix = y13;
			break;
		default:
			break;
	}
}

void backward (LAYER* layer, data* negdata, int* batch_chan)
{
	switch (layer->type)
	{
		case 1: ;
			CONV* conv = (CONV*) layer->layer;
			gsl_matrix*** y1 = negdata->image;
			gsl_matrix*** x1 = backward_conv(conv, y1);
			for (int i = 0; i < conv->batch_size; i++)
			{
				for (int j = 0; j < *batch_chan; j++) gsl_matrix_free(negdata->image[i][j]);
				free(negdata->image[i]);
			}
			free(negdata->image);
			negdata->image = x1;
			*batch_chan = conv->n_channels;
			break;
		case 2: ;
			POOL* pool = (POOL*) layer->layer;
			gsl_matrix*** y2 = negdata->image;
			gsl_matrix*** x2 = backward_pool(pool, y2);
			for (int i = 0; i < pool->batch_size; i++)
			{
				for (int j = 0; j < *batch_chan; j++) gsl_matrix_free(negdata->image[i][j]);
				free(negdata->image[i]);
			}
			free(negdata->image);
			negdata->image = x2;
			*batch_chan = pool->n_channels;
			break;
		case 3: ;
			RELU* relu = (RELU*) layer->layer;
			gsl_matrix*** y3 = negdata->image;
			gsl_matrix*** x3 = backward_relu(relu, y3);
			for (int i = 0; i < relu->batch_size; i++)
			{
				for (int j = 0; j < *batch_chan; j++) gsl_matrix_free(negdata->image[i][j]);
				free(negdata->image[i]);
			}
			free(negdata->image);
			negdata->image = x3;
			*batch_chan = relu->n_channels;
			break;
		case 4: ;
			FLAT* flat = (FLAT*) layer->layer;
			gsl_matrix* y4 = negdata->matrix;
			gsl_matrix*** x4 = backward_flat(flat, y4);
			gsl_matrix_free(negdata->matrix);
			negdata->image = x4;
			*batch_chan = flat->n_channels;
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
		case 12: ;
			TANH* tanh = (TANH*) layer->layer;
			gsl_matrix* y12 = negdata->matrix;
			gsl_matrix* x12 = backward_tanh(tanh, y12);
			gsl_matrix_free(negdata->matrix);
			negdata->matrix = x12;
			break;
		case 13: ;
			RBML* rbml = (RBML*) layer->layer;
			gsl_matrix* y13 = negdata->matrix;
			gsl_matrix* x13 = backward_rbml(rbml, y13);
			gsl_matrix_free(negdata->matrix);
			negdata->matrix = x13;
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
		case 8:
			get_updates_relv((RELV*) layer->layer, learning_rate);
			break;
		case 9:
			get_updates_sigm((SIGM*) layer->layer, learning_rate);
			break;
		case 11:
			get_updates_dire((DIRE*) layer->layer, learning_rate);
			break;
		case 12:
			get_updates_tanh((TANH*) layer->layer, learning_rate);
			break;
		case 13:
			get_updates_rbml((RBML*) layer->layer, learning_rate);
			break;
		default:
			break;
	}
}

gsl_matrix* evaluate_loss (LAYER* layer, gsl_matrix* output, gsl_matrix* targets,
			   double learning_rate, double* loss, double* accl)
{
	gsl_matrix* results = NULL;
	switch (layer->type)
	{
		case 7:
			results = evaluate_xent((XENT*) layer->layer, output, targets, learning_rate, loss, accl);
			break;
		case 13:
			results = evaluate_rbml((RBML*) layer->layer, output, targets, learning_rate, loss, accl);
			break;
		default:
			break;
	}
	return results;
}

/*----------------------------------------------------------------------------*/
/* HOW TO TRAIN YOUR CNN                                                      */
/*----------------------------------------------------------------------------*/

// Function to train the CNN
//  param training_x      : loaded dataset (rows = examples, cols = features)
//  param training_y      : loaded labels (binarized vector into rows = examples, cols = labels)
//  param num_samples     : number of image samples
//  param num_channels    : number of channels per image
//  param layers          : array of layers
//  param num_layers      : number of layers in array
//  param loss_layer      : evaluation layer
//  param training_epochs : number of epochs used for training
//  param batch_size      : size of a batch used to train the CNN
//  param learning_rate   : learning rate used for training the CNN
//  param momentum        : momentum rate used for training the CNN (Currently not used)
//  param rand_seed       : random seed for training
//  returns               : average loss in minibatches
double train_cnn (gsl_matrix*** training_x, gsl_matrix* training_y, int num_samples,
	int num_channels, LAYER* layers, int num_layers, LAYER* loss_layer, int training_epochs,
	int batch_size, double learning_rate, double momentum, int rand_seed)
{
	srand(rand_seed);

	int num_batches = num_samples / batch_size;

	int img_h = training_x[0][0]->size1;
	int img_w = training_x[0][0]->size2;

	int out_size = training_y->size2;

	data batchdata;
	int batch_chan;

	double acc_loss = 0;
	double acc_class = 0;

	for (int epoch = 0; epoch < training_epochs; epoch++)
	{
		acc_loss = 0;
		acc_class = 0;

		for (int idx_ini = 0; idx_ini < num_samples - batch_size + 1; idx_ini += batch_size)
		{
			// Select mini_batch
			gsl_matrix*** minibatch = (gsl_matrix***) malloc(batch_size * sizeof(gsl_matrix**));
			for (int b = 0, idx = idx_ini; b < batch_size; b++, idx++)
			{
				minibatch[b] = (gsl_matrix**) malloc(num_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < num_channels; c++)
				{
					minibatch[b][c] = gsl_matrix_alloc(img_h, img_w);
					gsl_matrix_memcpy(minibatch[b][c], training_x[idx][c]);
				}
			}
			
			gsl_matrix* targets = NULL;
			if (training_y)
			{
				targets = gsl_matrix_alloc(batch_size, out_size);
				for (int b = 0, idx = idx_ini; b < batch_size; b++, idx++)
				{
					gsl_vector* aux = gsl_vector_alloc(out_size);
					gsl_matrix_get_row(aux, training_y, idx);
					gsl_matrix_set_row(targets, b, aux);
					gsl_vector_free(aux);
				}
			}
			
			// Forward through layers
			batchdata.image = minibatch;
			batch_chan = num_channels;
			for (int i = 0; i < num_layers; i++)
				forward(&(layers[i]), &batchdata, &batch_chan);
			gsl_matrix* output = batchdata.matrix;

			// Calculate Loss
			double loss = 0, accl = 0;
			gsl_matrix* results = evaluate_loss(loss_layer, output, targets, learning_rate, &loss, &accl);
			acc_loss += loss;
			acc_class += accl;
			gsl_matrix_free(output);

			// Backward through layers, and update them
			batchdata.matrix = results;
			batch_chan = 0;
			for (int i = num_layers - 1; i >= 0; i--)
			{
				backward(&(layers[i]), &batchdata, &batch_chan);
				get_updates(&(layers[i]), learning_rate);
			}
			
			// Clean structures
			for (int i = 0; i < batch_size; i++)
			{
				for (int j = 0; j < num_channels; j++) gsl_matrix_free(batchdata.image[i][j]);
				free(batchdata.image[i]);
			}
			free(batchdata.image);
		}
//		if (epoch % 1 == 0)
			printf("Epoch %d: Mean Loss %f, Classification Accuracy %f\n", epoch, acc_loss / num_batches, acc_class / num_batches);
	}
	return (acc_loss / num_batches);
}

/*----------------------------------------------------------------------------*/
/* PREDICTION USING THE CNN                                                   */
/*----------------------------------------------------------------------------*/

// Function to predict the results of a matrix
gsl_matrix* prediction_cnn (gsl_matrix*** testing_x, int num_samples,
	int num_channels, LAYER* layers, int num_layers, int batch_size)
{
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
	else if (layers[num_layers - 1].type == 12) // TANH
		num_outputs = ((TANH*)(layers[num_layers - 1].layer))->n_units;
	else if (layers[num_layers - 1].type == 13) // RBML
		num_outputs = ((RBML*)(layers[num_layers - 1].layer))->n_hidden;

	gsl_matrix* result = gsl_matrix_alloc(num_samples, num_outputs);

	int img_h = testing_x[0][0]->size1;
	int img_w = testing_x[0][0]->size2;

	data batchdata;
	int batch_chan;

	// Loop through examples
	for (int idx_ini = 0; idx_ini < num_samples; idx_ini += batch_size)
	{
		// Uneven rows are considered (not like in training)
		int real_batch_size = batch_size;
		if (idx_ini + batch_size > num_samples) real_batch_size = num_samples % batch_size;

		// Select mini_batch
		gsl_matrix*** minibatch = (gsl_matrix***) malloc(batch_size * sizeof(gsl_matrix**));
		for (int b = 0, idx = idx_ini; b < real_batch_size; b++, idx++)
		{
			minibatch[b] = (gsl_matrix**) malloc(num_channels * sizeof(gsl_matrix*));
			for (int c = 0; c < num_channels; c++)
			{
				minibatch[b][c] = gsl_matrix_alloc(img_h, img_w);
				gsl_matrix_memcpy(minibatch[b][c], testing_x[idx][c]);
			}
		}

		// Complete the uneven Mini-Batch
		if (batch_size > real_batch_size)
			for (int b = real_batch_size ; b < batch_size; b++)
			{
				minibatch[b] = (gsl_matrix**) malloc(num_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < num_channels; c++)
					minibatch[b][c] = gsl_matrix_calloc(img_h, img_w);
			}

		// Forward through layers
		batchdata.image = minibatch;
		batch_chan = num_channels;
		for (int i = 0; i < num_layers; i++)
			forward(&(layers[i]), &batchdata, &batch_chan);
		gsl_matrix* output = batchdata.matrix;

		// Add output to results
		for (int b = 0, idx = idx_ini; b < real_batch_size; b++, idx++)
		{
			gsl_vector* aux = gsl_vector_alloc(num_outputs);
			gsl_matrix_get_row(aux, output, b);
			gsl_matrix_set_row(result, idx, aux);
			gsl_vector_free(aux);
		}

		gsl_matrix_free(output);
	}

	return result;
}

// Function to produce output features and rebuild inputs
void pass_through_cnn (gsl_matrix*** testing_x, int num_samples,
	int num_channels, LAYER* layers, int num_layers, int batch_size,
	gsl_matrix** features, gsl_matrix**** rebuild)
{
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
	else if (layers[num_layers - 1].type == 12) // TANH
		num_outputs = ((TANH*)(layers[num_layers - 1].layer))->n_units;
	else if (layers[num_layers - 1].type == 13) // RBML
		num_outputs = ((RBML*)(layers[num_layers - 1].layer))->n_hidden;

	int img_h = testing_x[0][0]->size1;
	int img_w = testing_x[0][0]->size2;

	(*features) = gsl_matrix_alloc(num_samples, num_outputs);
	(*rebuild) = (gsl_matrix***) malloc(num_samples * sizeof(gsl_matrix**));

	data batchdata;
	int batch_chan;

	// Loop through examples
	for (int idx_ini = 0; idx_ini < num_samples; idx_ini += batch_size)
	{
		// Uneven rows are considered (not like in training)
		int real_batch_size = batch_size;
		if (idx_ini + batch_size > num_samples) real_batch_size = num_samples % batch_size;

		// Select mini_batch
		gsl_matrix*** minibatch = (gsl_matrix***) malloc(batch_size * sizeof(gsl_matrix**));
		for (int b = 0, idx = idx_ini; b < real_batch_size; b++, idx++)
		{
			minibatch[b] = (gsl_matrix**) malloc(num_channels * sizeof(gsl_matrix*));
			for (int c = 0; c < num_channels; c++)
			{
				minibatch[b][c] = gsl_matrix_alloc(img_h, img_w);
				gsl_matrix_memcpy(minibatch[b][c], testing_x[idx][c]);
			}
		}

		// Complete the uneven Mini-Batch
		if (batch_size > real_batch_size)
			for (int b = real_batch_size ; b < batch_size; b++)
			{
				minibatch[b] = (gsl_matrix**) malloc(num_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < num_channels; c++)
					minibatch[b][c] = gsl_matrix_calloc(img_h, img_w);
			}

		// Forward through layers
		batchdata.image = minibatch;
		batch_chan = num_channels;
		for (int i = 0; i < num_layers; i++)
			forward(&(layers[i]), &batchdata, &batch_chan);

		// Add output to features
		for (int b = 0, idx = idx_ini; b < real_batch_size; b++, idx++)
		{
			gsl_vector* aux = gsl_vector_alloc(num_outputs);
			gsl_matrix_get_row(aux, batchdata.matrix, b);
			gsl_matrix_set_row((*features), idx, aux);
			gsl_vector_free(aux);
		}

		// Backward through layers
		batch_chan = 0;
		for (int i = num_layers - 1; i >= 0; i--)
			backward(&(layers[i]), &batchdata, &batch_chan);

		// Add reinput to rebuild
		for (int b = 0, idx = idx_ini; b < real_batch_size; b++, idx++)
		{
			(*rebuild)[idx] = (gsl_matrix**) malloc(num_channels * sizeof(gsl_matrix*));
			for (int c = 0; c < num_channels; c++)
			{
				(*rebuild)[idx][c] = gsl_matrix_alloc(img_h, img_w);
				gsl_matrix_memcpy((*rebuild)[idx][c], batchdata.image[b][c]);
				gsl_matrix_free(batchdata.image[b][c]);
			}
			free(batchdata.image[b]);
			
		}
		free(batchdata.image);
	}

	return;
}
