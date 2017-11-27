/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including MultiLayer Perceptron Function Implementations

#include "cnn.h"

// Additional functions to manage forward and backward propagation
// are found in cnn.c file

/*---------------------------------------------------------------------------*/
/* HOW TO TRAIN YOUR MLP                                                     */
/*---------------------------------------------------------------------------*/

// Function to train a MLP FFANN, with a 2D matrix input
//  param training_x      : loaded dataset (rows = examples, cols = features)
//  param training_y      : loaded labels (binarized vector into rows = examples, cols = labels)
//  param layers          : array of layers
//  param num_layers      : number of layers in array
//  param training_epochs : number of epochs used for training
//  param batch_size      : size of a batch used to train the MLP
//  param learning_rate   : learning rate used for training the MLP
//  param momentum        : momentum rate used for training the MLP (Currently not used)
//  param rand_seed       : random seed for training
//  returns               : average loss in minibatches
double train_mlp (gsl_matrix* training_x, gsl_matrix* training_y, LAYER* layers,
	int num_layers, int training_epochs, int batch_size, double learning_rate,
	double momentum, int rand_seed)
{
	srand(rand_seed);

	int num_samples = training_x->size1;
	int num_columns = training_x->size2;
	int num_batches = num_samples / batch_size;

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
//			if (j % 20 == 0) printf("Batch number %d\n", j);

			// Select mini_batch
			int idx_ini = j * batch_size;
			int idx_fin = idx_ini + batch_size - 1;

			if (idx_fin >= num_samples) break;

			gsl_matrix* minibatch = gsl_matrix_alloc(batch_size, num_columns);
			gsl_matrix* targets = gsl_matrix_alloc(batch_size, out_size);
			for (int b = 0; b < batch_size; b++)
			{
				gsl_vector* aux1 = gsl_vector_alloc(num_columns);
				gsl_matrix_get_row(aux1, training_x, idx_ini + b);
				gsl_matrix_set_row(minibatch, b, aux1);
				gsl_vector_free(aux1);

				gsl_vector* aux2 = gsl_vector_alloc(out_size);
				gsl_matrix_get_row(aux2, training_y, idx_ini + b);
				gsl_matrix_set_row(targets, b, aux2);
				gsl_vector_free(aux2);
			}

			// Forward through layers
			batchdata.matrix = minibatch;
			for (int i = 0; i < num_layers; i++)
				forward(&(layers[i]), &batchdata);

			// Calculate Forward Loss and Negdata
			gsl_matrix* output = batchdata.matrix;
			gsl_matrix* pred_y = forward_cell(&loss_layer, output, targets);
			gsl_matrix* results = backward_cell(&loss_layer, output, targets);

			acc_loss += loss_layer.loss;
			acc_class += classification_accuracy(pred_y, targets);

			gsl_matrix_free(pred_y);
			gsl_matrix_free(output);
			gsl_matrix_free(targets);

			// Backward through layers
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
/* PREDICTION USING THE MLP                                                  */
/*---------------------------------------------------------------------------*/

// Function to predict the results of a matrix
gsl_matrix* prediction_mlp (gsl_matrix* testing_x, LAYER* layers, int num_layers, int batch_size)
{
	int num_samples = (int) testing_x->size1;
	int num_columns = (int) testing_x->size2;

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

	gsl_matrix* result = gsl_matrix_alloc(num_samples, num_outputs);

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

		gsl_matrix* minibatch = gsl_matrix_alloc(batch_size, num_columns);
		for (int b = 0; b < real_batch_size; b++)
		{
			gsl_vector* aux1 = gsl_vector_alloc(num_columns);
			gsl_matrix_get_row(aux1, testing_x, idx_ini + b);
			gsl_matrix_set_row(minibatch, b, aux1);
			gsl_vector_free(aux1);
		}

		// Completar el Mini-Batch
		if (batch_size > real_batch_size)
			for (int b = real_batch_size; b < batch_size; b++)
			{
				gsl_vector* aux1 = gsl_vector_alloc(num_columns);
				gsl_matrix_get_row(aux1, testing_x, idx_ini + b);
				gsl_matrix_set_row(minibatch, b, aux1);
				gsl_vector_free(aux1);
			}

		// Forward through layers
		batchdata.matrix = minibatch;
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

