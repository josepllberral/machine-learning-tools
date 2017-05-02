/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including R interface for CNNs
// Compile using "R CMD SHLIB cnn_r.c cell.c flat.c line.c matrix_ops.c msel.c relu.c sigm.c test.c cnn.c conv.c grad_check.c mlp.c pool.c relv.c soft.c -lgsl -lgslcblas -lm -o cnn.so"

#include "cnn.h"

#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>

/*---------------------------------------------------------------------------*/
/* INTERFACE TO R                                                            */
/*---------------------------------------------------------------------------*/

#define RARRAY(m,i,j,k,l) (REAL(m)[ INTEGER(GET_DIM(m))[0]*(l) + INTEGER(GET_DIM(m))[1]*(l)*(k) + INTEGER(GET_DIM(m))[2]*(l)*(k)*(j) + (i) ])

#define RMATRIX(m,i,j) (REAL(m)[ INTEGER(GET_DIM(m))[0]*(j)+(i) ])
#define RVECTOR(v,i) (REAL(v)[(i)])
#define RVECTORI(v,i) (INTEGER(v)[(i)])

LAYER* build_pipeline (SEXP layers, int nlays)
{
	LAYER* retval = (LAYER*) malloc(nlays * sizeof(LAYER));
	int layer_p = 0;
	for (int i = 0; i < nlays; i++)
	{
		SEXP laux = VECTOR_ELT(layers, i);
		if (strcmp(CHAR(STRING_ELT(laux, 0)), "CONV") == 0)
		{
			CONV* conv = (CONV*) malloc(sizeof(CONV));
			
			int nchan_conv = atoi(CHAR(STRING_ELT(laux, 1))); // n_channels
			int nfilt_conv = atoi(CHAR(STRING_ELT(laux, 2))); // n_filters
			int fsize_conv = atoi(CHAR(STRING_ELT(laux, 3))); // filter_size
			double scale_conv = atof(CHAR(STRING_ELT(laux, 4))); // scale
			int bmode_conv = atoi(CHAR(STRING_ELT(laux, 5))); // border_mode
			int bsize_conv = atoi(CHAR(STRING_ELT(laux, 6))); // batch_size

			create_CONV(conv, nchan_conv, nfilt_conv, fsize_conv, scale_conv, bmode_conv, bsize_conv);
			retval[layer_p].type = 1; retval[layer_p++].layer = (void*) conv;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "POOL") == 0)
		{
			POOL* pool = (POOL*) malloc(sizeof(POOL));

			int nchan_pool = atoi(CHAR(STRING_ELT(laux, 1))); // n_channels
			double scale_pool = atof(CHAR(STRING_ELT(laux, 2))); // scale
			int bsize_pool = atoi(CHAR(STRING_ELT(laux, 3))); // batch_size
			int wsize_pool = atoi(CHAR(STRING_ELT(laux, 4))); // win_size
			int strid_pool = atoi(CHAR(STRING_ELT(laux, 5))); // stride

			create_POOL(pool, nchan_pool, scale_pool, bsize_pool, wsize_pool, strid_pool);
			retval[layer_p].type = 2; retval[layer_p++].layer = (void*) pool;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "RELU") == 0)
		{
			RELU* relu = (RELU*) malloc(sizeof(RELU));

			int nchan_relu = atoi(CHAR(STRING_ELT(laux, 1))); // n_channels
			int bsize_relu = atoi(CHAR(STRING_ELT(laux, 2))); // batch_size

			create_RELU(relu, nchan_relu, bsize_relu);
			retval[layer_p].type = 3; retval[layer_p++].layer = (void*) relu;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "FLAT") == 0)
		{
			FLAT* flat = (FLAT*) malloc(sizeof(FLAT));

			int nchan_flat = atoi(CHAR(STRING_ELT(laux, 1))); // n_channels
			int bsize_flat = atoi(CHAR(STRING_ELT(laux, 2))); // batch_size

			create_FLAT(flat, nchan_flat, bsize_flat);
			retval[layer_p].type = 4; retval[layer_p++].layer = (void*) flat;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "LINE") == 0)
		{
			LINE* line = (LINE*) malloc(sizeof(LINE));

			int nvis_line = atoi(CHAR(STRING_ELT(laux, 1))); // n_visible
			int nhid_line = atoi(CHAR(STRING_ELT(laux, 2))); // n_visible
			double scale_line = atof(CHAR(STRING_ELT(laux, 3))); // scale
			int bsize_line = atoi(CHAR(STRING_ELT(laux, 4))); // batch_size

			create_LINE(line, nvis_line, nhid_line, scale_line, bsize_line);
			retval[layer_p].type = 5; retval[layer_p].layer = (void*) line;
			layer_p++;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "SOFT") == 0)
		{
			SOFT* soft = (SOFT*) malloc(sizeof(SOFT));

			int nunits_soft = atoi(CHAR(STRING_ELT(laux, 1))); // n_units
			int bsize_soft = atoi(CHAR(STRING_ELT(laux, 2))); // batch_size

			create_SOFT(soft, nunits_soft, bsize_soft);
			retval[layer_p].type = 6; retval[layer_p++].layer = (void*) soft;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "RELV") == 0)
		{
			RELV* relv = (RELV*) malloc(sizeof(RELV));

			int bsize_relv = atoi(CHAR(STRING_ELT(laux, 1))); // batch_size

			create_RELV(relv, bsize_relv);
			retval[layer_p].type = 8; retval[layer_p++].layer = (void*) relv;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "SIGM") == 0)
		{
			SIGM* sigm = (SIGM*) malloc(sizeof(SIGM));

			int nunits_sigm = atoi(CHAR(STRING_ELT(laux, 1))); // n_units
			int bsize_sigm = atoi(CHAR(STRING_ELT(laux, 2))); // batch_size

			create_SIGM(sigm, nunits_sigm, bsize_sigm);
			retval[layer_p].type = 9; retval[layer_p++].layer = (void*) sigm;
		}
	}

	return retval;
}

void free_pipeline (LAYER* layers, int nlays)
{
	for (int i = 0; i < nlays; i++)
	{
		if (layers[i].type == 1) free_CONV((CONV*) layers[i].layer);
		else if (layers[i].type == 2) free_POOL((POOL*) layers[i].layer);
		else if (layers[i].type == 3) free_RELU((RELU*) layers[i].layer);
		else if (layers[i].type == 4) free_FLAT((FLAT*) layers[i].layer);
		else if (layers[i].type == 5) free_LINE((LINE*) layers[i].layer);
		else if (layers[i].type == 6) free_SOFT((SOFT*) layers[i].layer);
		else if (layers[i].type == 8) free_RELV((RELV*) layers[i].layer);
		else if (layers[i].type == 10) free_SIGM((SIGM*) layers[i].layer);
	}
	return;
}


// Interface for Training a CRBM
SEXP _C_CNN_train (SEXP dataset, SEXP targets, SEXP layers, SEXP num_layers, SEXP batch_size,
	SEXP training_epochs, SEXP learning_rate, SEXP momentum, SEXP rand_seed)
{
 	int nrows = INTEGER(GET_DIM(dataset))[0];
 	int nchan = INTEGER(GET_DIM(dataset))[1];
 	int img_h = INTEGER(GET_DIM(dataset))[2];
 	int img_w = INTEGER(GET_DIM(dataset))[3];

 	int nouts = INTEGER(GET_DIM(targets))[1];

 	int basi = INTEGER_VALUE(batch_size);
 	int trep = INTEGER_VALUE(training_epochs);
 	int rase = INTEGER_VALUE(rand_seed);
 	double lera = NUMERIC_VALUE(learning_rate);
 	double mome = NUMERIC_VALUE(momentum);

	int nlays = INTEGER_VALUE(num_layers);

printf("Read Datasets\n");
	// Create Dataset Structure
	gsl_matrix*** train_X = (gsl_matrix***) malloc(nrows * sizeof(gsl_matrix**));
	gsl_matrix* train_Y = gsl_matrix_alloc(nrows, nouts);
	for (int r = 0; r < nrows; r++)
	{
		train_X[r] = (gsl_matrix**) malloc(nchan * sizeof(gsl_matrix*));
		for (int c = 0; c < nchan; c++)
		{
			train_X[r][c] = gsl_matrix_alloc(img_h, img_w);
			for (int h = 0; h < img_h; h++)
				for (int w = 0; w < img_w; w++)
					gsl_matrix_set(train_X[r][c], h, w, RARRAY(dataset, r, c, h, w));
		}
		for (int o = 0; o < nouts; o++)
			gsl_matrix_set(train_Y, r, o, RMATRIX(targets, r, o));
	}

printf("Build Pipeline\n");

	// Build the Layers pipeline
	LAYER* pipeline = build_pipeline(layers, nlays);

printf("Start Training\n");
	// Train a CNN
	double loss = train_cnn (train_X, train_Y, nrows, nchan, pipeline, nlays, trep, basi, lera, mome, rase);

//--------------------------------

	// Return Structure
//	SEXP retval = PROTECT(allocVector(VECSXP, nlays));

	// TODO - ...

//	SEXP nms = PROTECT(allocVector(STRSXP, nlays));

//	UNPROTECT(2);

//--------------------------------

printf("Freeing Elements\n");

	free_pipeline (pipeline, nlays);

	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < nchan; j++)
			gsl_matrix_free(train_X[i][j]);
		free(train_X[i]);
	}
	free(train_X);
	gsl_matrix_free(train_Y);

printf("Done...\n");

SEXP retval = PROTECT(allocVector(VECSXP, 1));
SET_VECTOR_ELT(retval, 0, allocVector(INTSXP, 1));
INTEGER(VECTOR_ELT(retval, 0))[0] = 10;
UNPROTECT(1);

printf("Returning\n");
	return retval;
}
/*
	// Return Structure
	SEXP retval = PROTECT(allocVector(VECSXP, 14));

	SET_VECTOR_ELT(retval, 0, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 0))[0] = crbm.N;

	SET_VECTOR_ELT(retval, 1, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 1))[0] = crbm.n_visible;

	SET_VECTOR_ELT(retval, 2, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 2))[0] = crbm.n_hidden;

	SET_VECTOR_ELT(retval, 3, allocMatrix(REALSXP, ncol, nhid));
	SET_VECTOR_ELT(retval, 7, allocVector(REALSXP, ncol));
	SET_VECTOR_ELT(retval, 8, allocMatrix(REALSXP, ncol, nhid));
	SET_VECTOR_ELT(retval, 12, allocVector(REALSXP, ncol));
	for (int i = 0; i < ncol; i++)
	{
		for (int j = 0; j < nhid; j++)
		{
			REAL(VECTOR_ELT(retval, 3))[i * nhid + j] = gsl_matrix_get(crbm.W, i, j);
			REAL(VECTOR_ELT(retval, 8))[i * nhid + j] = gsl_matrix_get(crbm.vel_W, i, j);
		}
		REAL(VECTOR_ELT(retval, 7))[i] = gsl_vector_get(crbm.vbias, i);
		REAL(VECTOR_ELT(retval, 12))[i] = gsl_vector_get(crbm.vel_v, i);
	}

	SET_VECTOR_ELT(retval, 4, allocMatrix(REALSXP, ncol * dely, nhid));
	SET_VECTOR_ELT(retval, 5, allocMatrix(REALSXP, ncol * dely, ncol));
	SET_VECTOR_ELT(retval, 9, allocMatrix(REALSXP, ncol * dely, nhid));
	SET_VECTOR_ELT(retval, 10, allocMatrix(REALSXP, ncol * dely, ncol));
	for (int i = 0; i < ncol * dely; i++)
	{
		for (int j = 0; j < nhid; j++)
		{
			REAL(VECTOR_ELT(retval, 4))[i * nhid + j] = gsl_matrix_get(crbm.B, i, j);
			REAL(VECTOR_ELT(retval, 9))[i * nhid + j] = gsl_matrix_get(crbm.vel_B, i, j);
		}
		for (int j = 0; j < ncol; j++)
		{
			REAL(VECTOR_ELT(retval, 5))[i * ncol + j] = gsl_matrix_get(crbm.A, i, j);
			REAL(VECTOR_ELT(retval, 10))[i * ncol + j] = gsl_matrix_get(crbm.vel_A, i, j);
		}
	}

	SET_VECTOR_ELT(retval, 6, allocVector(REALSXP, nhid));
	SET_VECTOR_ELT(retval, 11, allocVector(REALSXP, nhid));
	for (int i = 0; i < nhid; i++)
	{
		REAL(VECTOR_ELT(retval, 6))[i] = gsl_vector_get(crbm.hbias, i);
		REAL(VECTOR_ELT(retval, 11))[i] = gsl_vector_get(crbm.vel_h, i);
	}

	SET_VECTOR_ELT(retval, 13, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 13))[0] = crbm.delay;

	SEXP nms = PROTECT(allocVector(STRSXP, 14));
	SET_STRING_ELT(nms, 0, mkChar("N"));
	SET_STRING_ELT(nms, 1, mkChar("n_visible"));
	SET_STRING_ELT(nms, 2, mkChar("n_hidden"));
	SET_STRING_ELT(nms, 3, mkChar("W"));
	SET_STRING_ELT(nms, 4, mkChar("B"));
	SET_STRING_ELT(nms, 5, mkChar("A"));
	SET_STRING_ELT(nms, 6, mkChar("hbias"));
	SET_STRING_ELT(nms, 7, mkChar("vbias"));
	SET_STRING_ELT(nms, 8, mkChar("vel_W"));
	SET_STRING_ELT(nms, 9, mkChar("vel_B"));
	SET_STRING_ELT(nms, 10, mkChar("vel_A"));
	SET_STRING_ELT(nms, 11, mkChar("vel_h"));
	SET_STRING_ELT(nms, 12, mkChar("vel_v"));
	SET_STRING_ELT(nms, 13, mkChar("delay"));

	setAttrib(retval, R_NamesSymbol, nms);
	UNPROTECT(2);

	// Free Dataset Structure
	free(seq_len_p);
	free_CRBM(&crbm);

	gsl_matrix_free(train_X_p);

	return retval;
}*/

// Function to Re-assemble the CNN
void reassemble_CNN() { return; }
/*	(CRBM* crbm, SEXP W_input, SEXP B_input, SEXP A_input,
	SEXP hbias_input, SEXP vbias_input, int nhid, int nvis,	int dely)
{
 	int wrow = INTEGER(GET_DIM(W_input))[0];
 	int wcol = INTEGER(GET_DIM(W_input))[1];

	gsl_matrix* W = gsl_matrix_alloc(wrow, wcol);
	for (int i = 0; i < wrow; i++)
		for (int j = 0; j < wcol; j++)
			gsl_matrix_set(W, i, j, RMATRIX(W_input, i, j));

 	int brow = INTEGER(GET_DIM(B_input))[0];
 	int bcol = INTEGER(GET_DIM(B_input))[1];

	gsl_matrix* B = gsl_matrix_alloc(brow, bcol);
	for (int i = 0; i < brow; i++)
		for (int j = 0; j < bcol; j++)
			gsl_matrix_set(B, i, j, RMATRIX(B_input, i, j));

 	int arow = INTEGER(GET_DIM(A_input))[0];
 	int acol = INTEGER(GET_DIM(A_input))[1];

	gsl_matrix* A = gsl_matrix_alloc(arow, acol);
	for (int i = 0; i < arow; i++)
		for (int j = 0; j < acol; j++)
			gsl_matrix_set(A, i, j, RMATRIX(A_input, i, j));

	gsl_vector* hbias = gsl_vector_calloc(nhid);
	for (int i = 0; i < nhid; i++)
		gsl_vector_set(hbias, i, RVECTOR(hbias_input, i));

	gsl_vector* vbias = gsl_vector_calloc(nvis);
	for (int i = 0; i < nvis; i++)
		gsl_vector_set(vbias, i, RVECTOR(vbias_input, i));

	create_CRBM (crbm, 0, nvis, nhid, dely, 1, A, B, W, hbias, vbias);
}*/

// Interface for Predicting and Reconstructing using a CNN
SEXP _C_CNN_predict() { return NULL; }
/*	(SEXP newdata, SEXP n_visible, SEXP n_hidden, SEXP W_input,
	SEXP B_input, SEXP A_input, SEXP hbias_input, SEXP vbias_input, SEXP delay)
{
	int nrow = INTEGER(GET_DIM(newdata))[0];
 	int ncol = INTEGER(GET_DIM(newdata))[1];

	int nvis = INTEGER_VALUE(n_visible);
	int nhid = INTEGER_VALUE(n_hidden);
	int dely = INTEGER_VALUE(delay);

	// Re-assemble the CRBM
	CRBM crbm;
	reassemble_CRBM (&crbm, W_input, B_input, A_input, hbias_input, vbias_input,
		nhid, nvis, dely);

	// Prepare Test Dataset
	gsl_matrix* test_X_p = gsl_matrix_alloc(nrow, ncol);
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			gsl_matrix_set(test_X_p, i, j, RMATRIX(newdata, i, j));

	// Pass through CRBM
	gsl_matrix* reconstruction = gsl_matrix_calloc(nrow, ncol);
	gsl_matrix* activations = gsl_matrix_calloc(nrow, nhid);
	reconstruct_CRBM(&crbm, test_X_p, &activations, &reconstruction);

	// Prepare Results
	SEXP retval = PROTECT(allocVector(VECSXP, 2));

	SET_VECTOR_ELT(retval, 0, allocMatrix(REALSXP, nrow, ncol));
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			REAL(VECTOR_ELT(retval, 0))[i * ncol + j] = gsl_matrix_get(reconstruction, i, j);

	SET_VECTOR_ELT(retval, 1, allocMatrix(REALSXP, nrow, nhid));
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < nhid; j++)
			REAL(VECTOR_ELT(retval, 1))[i * nhid + j] = gsl_matrix_get(activations, i, j);

	SEXP nms = PROTECT(allocVector(STRSXP, 2));
	SET_STRING_ELT(nms, 0, mkChar("reconstruction"));
	SET_STRING_ELT(nms, 1, mkChar("activation"));

	setAttrib(retval, R_NamesSymbol, nms);
	UNPROTECT(2);

	// Free the structures and the CRBM
	free_CRBM(&crbm);

	gsl_matrix_free(reconstruction);
	gsl_matrix_free(activations);
	gsl_matrix_free(test_X_p);

	return retval;
}*/

