/*----------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C                                                */
/*----------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File with Neural Network builders (CNN and MLP)

#include "cnn.h"

/*----------------------------------------------------------------------------*/
/* AUXILIAR FUNCTIONS                                                         */
/*----------------------------------------------------------------------------*/

void read_images (gsl_matrix*** dataset, char* filename, int num_channels, int nrow, int ncol, int img_h, int img_w)
{
	FILE * fp;
	char * line = NULL;
	size_t len = 0;

	fp = fopen(filename, "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	for (int i = 0; i < nrow; i++)
	{
		dataset[i] = (gsl_matrix**) malloc(num_channels * sizeof(gsl_matrix*));
		for (int c = 0; c < num_channels; c++) // FIXME - Watch out the order "Height x Channel x Width"
		{
			dataset[i][c] = gsl_matrix_alloc(img_h, img_w);

			ssize_t read = getline(&line, &len, fp);
			char *ch = strtok(line, " ");
			for (int j = 0; j < ncol; j++)
			{
				int idx_h = j % img_h;
				int idx_w = j / img_w;
				gsl_matrix_set(dataset[i][c], idx_h, idx_w, atof(ch));
				ch = strtok(NULL, " ");
			}
			free(ch);
		}
	}
	fclose(fp);

	if (line) free(line);
}

void read_labels (gsl_matrix* labels, char* filename, int nrow)
{
	FILE * fp;
	char * line = NULL;
	size_t len = 0;

	fp = fopen(filename, "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	for (int i = 0; i < nrow; i++)
	{
		ssize_t read = getline(&line, &len, fp);
		int y = atoi(line);
		gsl_matrix_set(labels, i, y, 1.0);
	}
	fclose(fp);

	if (line) free(line);
}

void read_images_flat (gsl_matrix* dataset, char* filename, int nrow, int ncol)
{
	FILE * fp;
	char * line = NULL;
	size_t len = 0;

	fp = fopen(filename, "r");
	if (fp == NULL) exit(EXIT_FAILURE);

	for (int i = 0; i < nrow; i++)
	{
		ssize_t read = getline(&line, &len, fp);
		char *ch = strtok(line, " ");
		for (int j = 0; j < ncol; j++)
		{
			gsl_matrix_set(dataset, i, j, atof(ch));
			ch = strtok(NULL, " ");
		}
		free(ch);
	}
	fclose(fp);

	if (line) free(line);
}

/*----------------------------------------------------------------------------*/
/* MAIN FUNCTIONS                                                             */
/*----------------------------------------------------------------------------*/

int main_cnn (char* filename_tr, char* filename_label, char* filename_test, char* filename_label_ts,
			  int nrow, int ncol, int nrow_ts, int num_channels, int img_h, int img_w, int nlabels)
{
	// Read Train Data
	gsl_matrix*** training_x = (gsl_matrix***) malloc(nrow * sizeof(gsl_matrix**));
	read_images(training_x, filename_tr, num_channels, nrow, ncol, img_h, img_w);

	gsl_matrix* training_y = gsl_matrix_calloc(nrow, nlabels);
	read_labels(training_y, filename_label, nrow);

	printf("Training Dataset Read\n");

	// Read Test Data
	gsl_matrix*** testing_x = (gsl_matrix***) malloc(nrow_ts * sizeof(gsl_matrix**));
	read_images(testing_x, filename_test, num_channels, nrow_ts, ncol, img_h, img_w);

	gsl_matrix* testing_y = gsl_matrix_calloc(nrow_ts, nlabels);
	read_labels(testing_y, filename_label_ts, nrow_ts);
	
	printf("Testing Dataset Read\n");

	// Prepare the CNN for MNIST
	int batch_size = 10;
	int border_mode = 2;
	int filter_size = 5;

	int win_size = 3;
	int stride = 2;
	
	int num_layers = 11;
	int random_seed = 1234;

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
	
	XENT xent1; create_XENT(&xent1);
	LAYER* eval_layer = (LAYER*) malloc(sizeof(LAYER));
	eval_layer->type = 7; eval_layer->layer = (void*) &xent1;

	printf("CNN created\n");

	// Train a CNN to learn MNIST	
	int training_epochs = 15;
	double learning_rate = 1e-2;
	double momentum = 1;

	double loss = train_cnn (training_x, training_y, nrow, num_channels,
		layers, num_layers, eval_layer, training_epochs, batch_size, learning_rate, momentum, random_seed);

	printf("CNN trained\n");
	
	// TODO - Save the content of the Layers

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
	free_XENT(&xent1);

	free(layers);
	free(eval_layer);

	// Free the data
	for (int i = 0; i < nrow; i++)
	{
		for (int c = 0; c < num_channels; c++)
			gsl_matrix_free(training_x[i][c]);
		free(training_x[i]);
	}
	free(training_x);
	gsl_matrix_free(training_y);

	for (int i = 0; i < nrow_ts; i++)
	{
		gsl_matrix_free(testing_x[i][0]);
		free(testing_x[i]);
	}
	free(testing_x);
	gsl_matrix_free(testing_y);

	return 0;
}

int main_mlp (char* filename_tr, char* filename_label, char* filename_test, char* filename_label_ts,
			  int nrow, int ncol, int nrow_ts, int img_h, int img_w, int nlabels)
{
	// Read Train Data
	gsl_matrix* training_x = gsl_matrix_alloc(nrow, ncol);
	read_images_flat(training_x, filename_tr, nrow, ncol);

	gsl_matrix* training_y = gsl_matrix_calloc(nrow, nlabels);
	read_labels(training_y, filename_label, nrow);

	printf("Training Dataset Read\n");

	// Read Test Data
	gsl_matrix* testing_x = gsl_matrix_alloc(nrow_ts, ncol);
	read_images_flat(testing_x, filename_test, nrow_ts, ncol);

	gsl_matrix* testing_y = gsl_matrix_calloc(nrow_ts, nlabels);
	read_labels(testing_y, filename_label_ts, nrow_ts);

	printf("Testing Dataset Read\n");

	// Prepare the MLP for MNIST	
	int batch_size = 10;
	int random_seed = 1234;
	int num_layers = 4;
	
	LINE line1; create_LINE(&line1, 784, 64, 0.01, batch_size);
	RELV relv1; create_RELV(&relv1, batch_size);
	LINE line2; create_LINE(&line2, 64, 10, 0.1, batch_size);
	SOFT soft1; create_SOFT(&soft1, 10, batch_size);

	LAYER* layers = (LAYER*) malloc(num_layers * sizeof(LAYER));
	layers[0].type = 5; layers[0].layer = (void*) &line1;
	layers[1].type = 8; layers[1].layer = (void*) &relv1;
	layers[2].type = 5; layers[2].layer = (void*) &line2;
	layers[3].type = 6; layers[3].layer = (void*) &soft1;
	
	XENT xent1; create_XENT(&xent1);
	LAYER* eval_layer = (LAYER*) malloc(sizeof(LAYER));
	eval_layer->type = 7; eval_layer->layer = (void*) &xent1;

	printf("MLP created\n");

	// Train a MLP to learn MNIST
	int training_epochs = 10;
	double learning_rate = 1e-2;
	double momentum = 1;

	double loss = train_mlp (training_x, training_y, layers, num_layers, eval_layer,
		training_epochs, batch_size, learning_rate, momentum, random_seed);

	printf("MLP trained\n");
	
	// TODO - Save the content of the Layers

	// Free the Network
	free_LINE(&line1);
	free_RELV(&relv1);
	free_LINE(&line2);
	free_SOFT(&soft1);
	free_XENT(&xent1);

	free(layers);
	free(eval_layer);

	// Free the data
	gsl_matrix_free(training_x);
	gsl_matrix_free(training_y);

	gsl_matrix_free(testing_x);
	gsl_matrix_free(testing_y);

	return 0;
}

/*---------------------------------------------------------------------------*/
/* MAIN FUNCTION - TEST                                                      */
/*---------------------------------------------------------------------------*/

// This MAIN function trains a MNIST in a MLP and a CNN
int main(int argc, char** argv)
{
	// MNIST Files
	
	char* filename_tr = argv[1];		// "../datasets/mnist_trainx.data";
	char* filename_label = argv[2];		// "../datasets/mnist_trainy.data";
	char* filename_ts = argv[3];		// "../datasets/mnist_testx.data";
	char* filename_label_ts = argv[4];	// "../datasets/mnist_testy.data";
	
	// MNIST Properties
	
	int nrow = 60000;
	int ncol = 784;
	
	int nrow_ts = 10000;

	int num_channels = 1;
	int img_h = 28;
	int img_w = 28;
	
	int nlabels = 10;

	// Launch MLP and CONV
	
	printf("Starting MLP version\n");
	main_mlp(filename_tr, filename_label, filename_ts, filename_label_ts, nrow, ncol, nrow_ts, img_h, img_w, nlabels);
	
	printf("Starting CNN version\n");
	main_cnn(filename_tr, filename_label, filename_ts, filename_label_ts, nrow, ncol, nrow_ts, num_channels, img_h, img_w, nlabels);
	
	return 0;
}
