/*----------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                          */
/*----------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including R interface for CNNs
// Compile using "R CMD SHLIB cnn_r.c dire.c flat.c line.c matrix_ops.c msel.c relu.c sigm.c test.c cnn.c conv.c grad_check.c mlp.c pool.c relv.c soft.c xent.c rbml.c -lgsl -lgslcblas -lm -o cnn.so"

#include "cnn.h"

#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>

/*----------------------------------------------------------------------------*/
/* INTERFACE TO R                                                             */
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/* PIPELINE AUXILIAR FUNCTIONS                                                */
/*----------------------------------------------------------------------------*/

#define RMATRIX(m,i,j) (REAL(m)[ INTEGER(GET_DIM(m))[0]*(j)+(i) ])
#define RVECTOR(v,i) (REAL(v)[(i)])
#define RVECTORI(v,i) (INTEGER(v)[(i)])
#define RARRAY(m,i,j,k,l) (REAL(m)[INTEGER(GET_DIM(m))[2] * INTEGER(GET_DIM(m))[1] * INTEGER(GET_DIM(m))[0] * (l) + INTEGER(GET_DIM(m))[1] * INTEGER(GET_DIM(m))[0] * (k) + INTEGER(GET_DIM(m))[0] * (j) + (i) ])

LAYER* build_pipeline (SEXP layers, int nlays, int batch_size)
{
	LAYER* retval = (LAYER*) malloc(nlays * sizeof(LAYER));
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

			create_CONV(conv, nchan_conv, nfilt_conv, fsize_conv, scale_conv, bmode_conv, batch_size);
			retval[i].type = 1; retval[i].layer = (void*) conv;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "POOL") == 0)
		{
			POOL* pool = (POOL*) malloc(sizeof(POOL));

			int nchan_pool = atoi(CHAR(STRING_ELT(laux, 1))); // n_channels
			double scale_pool = atof(CHAR(STRING_ELT(laux, 2))); // scale
			int wsize_pool = atoi(CHAR(STRING_ELT(laux, 3))); // win_size
			int strid_pool = atoi(CHAR(STRING_ELT(laux, 4))); // stride

			create_POOL(pool, nchan_pool, scale_pool, batch_size, wsize_pool, strid_pool);
			retval[i].type = 2; retval[i].layer = (void*) pool;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "RELU") == 0)
		{
			RELU* relu = (RELU*) malloc(sizeof(RELU));

			int nchan_relu = atoi(CHAR(STRING_ELT(laux, 1))); // n_channels

			create_RELU(relu, nchan_relu, batch_size);
			retval[i].type = 3; retval[i].layer = (void*) relu;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "FLAT") == 0)
		{
			FLAT* flat = (FLAT*) malloc(sizeof(FLAT));

			int nchan_flat = atoi(CHAR(STRING_ELT(laux, 1))); // n_channels

			create_FLAT(flat, nchan_flat, batch_size);
			retval[i].type = 4; retval[i].layer = (void*) flat;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "LINE") == 0)
		{
			LINE* line = (LINE*) malloc(sizeof(LINE));

			int nvis_line = atoi(CHAR(STRING_ELT(laux, 1)));     // n_visible
			int nhid_line = atoi(CHAR(STRING_ELT(laux, 2)));     // n_visible
			double scale_line = atof(CHAR(STRING_ELT(laux, 3))); // scale

			create_LINE(line, nvis_line, nhid_line, scale_line, batch_size);
			retval[i].type = 5; retval[i].layer = (void*) line;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "SOFT") == 0)
		{
			SOFT* soft = (SOFT*) malloc(sizeof(SOFT));

			int nunits_soft = atoi(CHAR(STRING_ELT(laux, 1))); // n_units

			create_SOFT(soft, nunits_soft, batch_size);
			retval[i].type = 6; retval[i].layer = (void*) soft;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "RELV") == 0)
		{
			RELV* relv = (RELV*) malloc(sizeof(RELV));

			create_RELV(relv, batch_size);
			retval[i].type = 8; retval[i].layer = (void*) relv;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "SIGM") == 0)
		{
			SIGM* sigm = (SIGM*) malloc(sizeof(SIGM));

			int nunits_sigm = atoi(CHAR(STRING_ELT(laux, 1))); // n_units

			create_SIGM(sigm, nunits_sigm, batch_size);
			retval[i].type = 9; retval[i].layer = (void*) sigm;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "DIRE") == 0)
		{
			DIRE* dire = (DIRE*) malloc(sizeof(DIRE));

			int nunits_dire = atoi(CHAR(STRING_ELT(laux, 1))); // n_units

			create_DIRE(dire, nunits_dire, batch_size);
			retval[i].type = 11; retval[i].layer = (void*) dire;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "TANH") == 0)
		{
			TANH* tanh = (TANH*) malloc(sizeof(TANH));

			int nunits_tanh = atoi(CHAR(STRING_ELT(laux, 1))); // n_units

			create_TANH(tanh, nunits_tanh, batch_size);
			retval[i].type = 12; retval[i].layer = (void*) tanh;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "RBML") == 0)
		{
			RBML* rbml = (RBML*) malloc(sizeof(RBML));

			int nvis_rbml = atoi(CHAR(STRING_ELT(laux, 1)));     // n_visible
			int nhid_rbml = atoi(CHAR(STRING_ELT(laux, 2)));     // n_hidden
			double scale_rbml = atof(CHAR(STRING_ELT(laux, 3))); // scale
			int n_gibbs_rbml = atof(CHAR(STRING_ELT(laux, 4)));  // n_gibbs

			create_RBML(rbml, nvis_rbml, nhid_rbml, scale_rbml, n_gibbs_rbml, batch_size);
			retval[i].type = 13; retval[i].layer = (void*) rbml;
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
		else if (layers[i].type == 9) free_SIGM((SIGM*) layers[i].layer);
		else if (layers[i].type == 11) free_DIRE((DIRE*) layers[i].layer);
		else if (layers[i].type == 12) free_TANH((TANH*) layers[i].layer);
		else if (layers[i].type == 13) free_RBML((RBML*) layers[i].layer);
		free(layers[i].layer);
	}
	free(layers);
	return;
}

void return_pipeline (SEXP* retval, LAYER* pipeline, int nlays)
{
	for (int i = 0; i < nlays; i++)
	{
		if (pipeline[i].type == 1) // CONV Layer
		{
			CONV* aux = (CONV*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 10));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("CONV"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->filter_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->n_filters;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3))[0] = aux->n_channels;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4, allocVector(INTSXP, 2));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4))[0] = aux->pad_y;
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4))[1] = aux->pad_x;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5, allocVector(INTSXP, 2));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5))[0] = aux->win_h;
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5))[1] = aux->win_w;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 6, allocVector(REALSXP, aux->n_filters));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 7, allocVector(REALSXP, aux->n_filters));
			for (int j = 0; j < aux->n_filters; j++)
			{
				REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 6))[j] = gsl_vector_get(aux->b, j);
				REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 7))[j] = gsl_vector_get(aux->grad_b, j);
			}

			SEXP dim;
			PROTECT(dim = Rf_allocVector(INTSXP, 4));
			INTEGER(dim)[0] = aux->n_filters;
			INTEGER(dim)[1] = aux->n_channels;
			INTEGER(dim)[2] = aux->filter_size;
			INTEGER(dim)[3] = aux->filter_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 8, allocArray(REALSXP, dim));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 9, allocArray(REALSXP, dim));
			for (int f = 0; f < aux->n_filters; f++)
				for (int c = 0; c < aux->n_channels; c++)
					for (int h = 0; h < aux->filter_size; h++)
						for (int w = 0; w < aux->filter_size; w++)
						{
							int idx = w + aux->filter_size * (h + aux->filter_size * (c + aux->n_channels * f));
							REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 8))[idx] = gsl_matrix_get(aux->W[f][c], h, w);
							REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 9))[idx] = gsl_matrix_get(aux->grad_W[f][c], h, w);
						}

			UNPROTECT(1);

			SEXP naux = PROTECT(allocVector(STRSXP, 10));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("filter_size"));
			SET_STRING_ELT(naux, 2, mkChar("n_filters"));
			SET_STRING_ELT(naux, 3, mkChar("n_channels"));
			SET_STRING_ELT(naux, 4, mkChar("padding"));
			SET_STRING_ELT(naux, 5, mkChar("win_size"));
			SET_STRING_ELT(naux, 6, mkChar("b"));
			SET_STRING_ELT(naux, 7, mkChar("grad_b"));
			SET_STRING_ELT(naux, 8, mkChar("W"));
			SET_STRING_ELT(naux, 9, mkChar("grad_W"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);
		}
		else if (pipeline[i].type == 2) // POOL Layer
		{
			POOL* aux = (POOL*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 5));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("POOL"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->n_channels;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->win_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3))[0] = aux->stride;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4))[0] = aux->padding;

			SEXP naux = PROTECT(allocVector(STRSXP, 5));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("n_channels"));
			SET_STRING_ELT(naux, 2, mkChar("win_size"));
			SET_STRING_ELT(naux, 3, mkChar("stride"));
			SET_STRING_ELT(naux, 4, mkChar("padding"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 3) // RELU Layer
		{
			RELU* aux = (RELU*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 2));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("RELU"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->n_channels;

			SEXP naux = PROTECT(allocVector(STRSXP, 2));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("n_channels"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 4) // FLAT Layer
		{
			FLAT* aux = (FLAT*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 3));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("FLAT"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->n_channels;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 2));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->img_h;
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[1] = aux->img_w;

			SEXP naux = PROTECT(allocVector(STRSXP, 3));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("n_channels"));
			SET_STRING_ELT(naux, 2, mkChar("img_dims"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 5) // LINE Layer
		{
			LINE* aux = (LINE*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 7));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("LINE"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->n_visible;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->n_hidden;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3, allocVector(REALSXP, aux->n_hidden));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4, allocVector(REALSXP, aux->n_hidden));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5, allocMatrix(REALSXP, aux->n_hidden, aux->n_visible));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 6, allocMatrix(REALSXP, aux->n_hidden, aux->n_visible));
			for (int j = 0; j < aux->n_hidden; j++)
			{
				for (int k = 0; k < aux->n_visible; k++)
				{
					REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5))[k * aux->n_hidden + j] = gsl_matrix_get(aux->W, j, k);
					REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 6))[k * aux->n_hidden + j] = gsl_matrix_get(aux->grad_W, j, k);
				}
				REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3))[j] = gsl_vector_get(aux->b, j);
				REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4))[j] = gsl_vector_get(aux->grad_b, j);
			}

			SEXP naux = PROTECT(allocVector(STRSXP, 7));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("n_visible"));
			SET_STRING_ELT(naux, 2, mkChar("n_hidden"));
			SET_STRING_ELT(naux, 3, mkChar("b"));
			SET_STRING_ELT(naux, 4, mkChar("grad_b"));
			SET_STRING_ELT(naux, 5, mkChar("W"));
			SET_STRING_ELT(naux, 6, mkChar("grad_W"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 6) // SOFT Layer
		{
			SOFT* aux = (SOFT*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 2));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("SOFT"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->n_units;

			SEXP naux = PROTECT(allocVector(STRSXP, 2));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("n_units"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 8) // RELV Layer
		{
			RELV* aux = (RELV*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 1));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("RELV"));

			SEXP naux = PROTECT(allocVector(STRSXP, 1));
			SET_STRING_ELT(naux, 0, mkChar("type"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 9) // SIGM Layer
		{
			SIGM* aux = (SIGM*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 2));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("SIGM"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->n_units;

			SEXP naux = PROTECT(allocVector(STRSXP, 2));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("n_units"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 11) // DIRE Layer
		{
			DIRE* aux = (DIRE*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 5));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("DIRE"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->batch_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->n_units;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3, allocMatrix(REALSXP, aux->batch_size, aux->n_units));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4, allocMatrix(REALSXP, aux->batch_size, aux->n_units));
			for (int j = 0; j < aux->batch_size; j++)
				for (int k = 0; k < aux->n_units; k++)
				{
					REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3))[k * aux->batch_size + j] = gsl_matrix_get(aux->buff_x, j, k);
					REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4))[k * aux->batch_size + j] = gsl_matrix_get(aux->buff_dy, j, k);
				}

			SEXP naux = PROTECT(allocVector(STRSXP, 5));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("batch_size"));
			SET_STRING_ELT(naux, 2, mkChar("n_units"));
			SET_STRING_ELT(naux, 3, mkChar("buffer_x"));
			SET_STRING_ELT(naux, 4, mkChar("buffer_dy"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 12) // TANH Layer
		{
			TANH* aux = (TANH*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 2));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("TANH"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->n_units;

			SEXP naux = PROTECT(allocVector(STRSXP, 2));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("n_units"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 13) // RBML Layer
		{
			RBML* aux = (RBML*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 10));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("RBML"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->n_visible;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->n_hidden;
			
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3))[0] = aux->n_gibbs;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4, allocVector(REALSXP, aux->n_hidden));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5, allocVector(REALSXP, aux->n_hidden));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 6, allocVector(REALSXP, aux->n_visible));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 7, allocVector(REALSXP, aux->n_visible));			
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 8, allocMatrix(REALSXP, aux->n_hidden, aux->n_visible));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 9, allocMatrix(REALSXP, aux->n_hidden, aux->n_visible));
			for (int j = 0; j < aux->n_hidden; j++)
			{
				for (int k = 0; k < aux->n_visible; k++)
				{
					REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 8))[k * aux->n_hidden + j] = gsl_matrix_get(aux->W, j, k); //CHECK!
					REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 9))[k * aux->n_hidden + j] = gsl_matrix_get(aux->grad_W, j, k); //CHECK!
				}
				REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4))[j] = gsl_vector_get(aux->hbias, j);
				REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5))[j] = gsl_vector_get(aux->grad_hbias, j);
			}
			for (int k = 0; k < aux->n_visible; k++)
			{
				REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 6))[k] = gsl_vector_get(aux->vbias, k);
				REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 7))[k] = gsl_vector_get(aux->grad_vbias, k);
			}

			SEXP naux = PROTECT(allocVector(STRSXP, 10));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("n_visible"));
			SET_STRING_ELT(naux, 2, mkChar("n_hidden"));
			SET_STRING_ELT(naux, 3, mkChar("n_gibbs"));
			SET_STRING_ELT(naux, 4, mkChar("hbias"));
			SET_STRING_ELT(naux, 5, mkChar("grad_hbias"));
			SET_STRING_ELT(naux, 6, mkChar("vbias"));
			SET_STRING_ELT(naux, 7, mkChar("grad_vbias"));
			SET_STRING_ELT(naux, 8, mkChar("W"));
			SET_STRING_ELT(naux, 9, mkChar("grad_W"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else // XENT, MSEL, others...
		{
			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 2));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("UNKN"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = 0;

			SEXP naux = PROTECT(allocVector(STRSXP, 2));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("dummy"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
	}
}

// Function to get elements by name from a R list
SEXP getListElement(SEXP list, const char *str)
{
	SEXP elmt = R_NilValue, names = getAttrib(list, R_NamesSymbol);

	for (R_len_t i = 0; i < length(list); i++)
		if (strcmp(CHAR(STRING_ELT(names, i)), str) == 0)
		{
			elmt = VECTOR_ELT(list, i);
			break;
		}
	return elmt;
}

// Function to Re-assemble the CNN or MLP
LAYER* reassemble_CNN (SEXP layers, int num_layers, int batch_size)
{
	LAYER* pipeline = (LAYER*) malloc(num_layers * sizeof(LAYER));

	for (int i = 0; i < num_layers; i++)
	{
		const char* s = CHAR(STRING_ELT(getListElement(VECTOR_ELT(layers, i), "type"),0));
		if (strcmp(s, "CONV") == 0)
		{
			CONV* aux = (CONV*) malloc(sizeof(CONV));

			aux->batch_size = batch_size;
			aux->filter_size = INTEGER(getListElement(VECTOR_ELT(layers, i), "filter_size"))[0];
			aux->n_filters = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_filters"))[0];
			aux->n_channels = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_channels"))[0];
			aux->win_h = INTEGER(getListElement(VECTOR_ELT(layers, i), "win_size"))[0];
			aux->win_w = INTEGER(getListElement(VECTOR_ELT(layers, i), "win_size"))[1];

			SEXP W = PROTECT(getListElement(VECTOR_ELT(layers, i), "W"));
			SEXP grad_W = PROTECT(getListElement(VECTOR_ELT(layers, i), "grad_W"));
			SEXP b = PROTECT(getListElement(VECTOR_ELT(layers, i), "b"));
			SEXP grad_b = PROTECT(getListElement(VECTOR_ELT(layers, i), "grad_b"));

			aux->W = (gsl_matrix***) malloc(aux->n_filters * sizeof(gsl_matrix**));
			aux->grad_W = (gsl_matrix***) malloc(aux->n_filters * sizeof(gsl_matrix**));
			aux->b = gsl_vector_calloc(aux->n_filters);
			aux->grad_b = gsl_vector_calloc(aux->n_filters);
			for (int f = 0; f < aux->n_filters; f++)
			{
				aux->W[f] = (gsl_matrix**) malloc(aux->n_channels * sizeof(gsl_matrix*));
				aux->grad_W[f] = (gsl_matrix**) malloc(aux->n_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < aux->n_channels; c++)
				{
					aux->W[f][c] = gsl_matrix_calloc(aux->filter_size, aux->filter_size);
					aux->grad_W[f][c] = gsl_matrix_calloc(aux->filter_size, aux->filter_size);
					for (int h = 0; h < aux->filter_size; h++)
						for (int w = 0; w < aux->filter_size; w++)
						{
							gsl_matrix_set(aux->W[f][c], h, w, RARRAY(W,f,c,h,w));
							gsl_matrix_set(aux->grad_W[f][c], h, w, RARRAY(grad_W,f,c,h,w));  
						}
				}
				gsl_vector_set(aux->b, f, RVECTOR(b, f));
				gsl_vector_set(aux->grad_b, f, RVECTOR(grad_b, f));

			}
			UNPROTECT(4);

			aux->pad_y = INTEGER(getListElement(VECTOR_ELT(layers, i), "padding"))[0];
			aux->pad_x = INTEGER(getListElement(VECTOR_ELT(layers, i), "padding"))[1];

			aux->img = (gsl_matrix***) malloc(aux->batch_size * sizeof(gsl_matrix**));
			for (int b = 0; b < aux->batch_size; b++)
			{
				aux->img[b] = (gsl_matrix**) malloc(aux->n_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < aux->n_channels; c++)
					aux->img[b][c] = gsl_matrix_calloc(1, 1);
			}

			pipeline[i].type = 1;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "POOL") == 0)
		{
			POOL* aux = (POOL*) malloc(sizeof(POOL));

			aux->batch_size = batch_size;
			aux->n_channels = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_channels"))[0];
			aux->win_size = INTEGER(getListElement(VECTOR_ELT(layers, i), "win_size"))[0];
			aux->stride = INTEGER(getListElement(VECTOR_ELT(layers, i), "stride"))[0];
			aux->padding = INTEGER(getListElement(VECTOR_ELT(layers, i), "padding"))[0];

			aux->img = (gsl_matrix***) malloc(aux->batch_size * sizeof(gsl_matrix**));
			for (int b = 0; b < aux->batch_size; b++)
			{
				aux->img[b] = (gsl_matrix**) malloc(aux->n_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < aux->n_channels; c++)
					aux->img[b][c] = gsl_matrix_calloc(1, 1);
			}

			pipeline[i].type = 2;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "RELU") == 0)
		{
			RELU* aux = (RELU*) malloc(sizeof(RELU));

			aux->batch_size = batch_size;
			aux->n_channels = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_channels"))[0];

			aux->img = (gsl_matrix***) malloc(aux->batch_size * sizeof(gsl_matrix**));
			for (int b = 0; b < aux->batch_size; b++)
			{
				aux->img[b] = (gsl_matrix**) malloc(aux->n_channels * sizeof(gsl_matrix*));
				for (int c = 0; c < aux->n_channels; c++)
					aux->img[b][c] = gsl_matrix_calloc(1, 1);
			}

			pipeline[i].type = 3;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "FLAT") == 0)
		{
			FLAT* aux = (FLAT*) malloc(sizeof(FLAT));

			aux->batch_size = batch_size;
			aux->n_channels = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_channels"))[0];
			aux->img_h = INTEGER(getListElement(VECTOR_ELT(layers, i), "img_dims"))[0];
			aux->img_w = INTEGER(getListElement(VECTOR_ELT(layers, i), "img_dims"))[1];

			pipeline[i].type = 4;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "LINE") == 0)
		{
			LINE* aux = (LINE*) malloc(sizeof(LINE));
			
			aux->batch_size = batch_size;
			aux->n_visible = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_visible"))[0];
			aux->n_hidden = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_hidden"))[0];

			SEXP W = PROTECT(getListElement(VECTOR_ELT(layers, i), "W"));
			SEXP grad_W = PROTECT(getListElement(VECTOR_ELT(layers, i), "grad_W"));
			SEXP b = PROTECT(getListElement(VECTOR_ELT(layers, i), "b"));
			SEXP grad_b = PROTECT(getListElement(VECTOR_ELT(layers, i), "grad_b"));

			aux->W = gsl_matrix_calloc(aux->n_hidden, aux->n_visible);
			aux->grad_W = gsl_matrix_calloc(aux->n_hidden, aux->n_visible);
			aux->b = gsl_vector_calloc(aux->n_hidden);
			aux->grad_b = gsl_vector_calloc(aux->n_hidden);

			for (int h = 0; h < aux->n_hidden; h++)
			{
				for (int v = 0; v < aux->n_visible; v++)
				{
					gsl_matrix_set(aux->W, h, v, RMATRIX(W,h,v));
					gsl_matrix_set(aux->grad_W, h, v, RMATRIX(grad_W,h,v));
				}
				gsl_vector_set(aux->b, h, RVECTOR(b, h));
				gsl_vector_set(aux->grad_b, h, RVECTOR(grad_b, h));

			}
			UNPROTECT(4);

			aux->x = gsl_matrix_calloc(aux->batch_size, aux->n_visible);

			pipeline[i].type = 5;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "SOFT") == 0)
		{
			SOFT* aux = (SOFT*) malloc(sizeof(SOFT));

			aux->batch_size = batch_size;
			aux->n_units = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_units"))[0];

			pipeline[i].type = 6;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "RELV") == 0)
		{
			RELV* aux = (RELV*) malloc(sizeof(RELV));

			aux->batch_size = batch_size;
			aux->img = gsl_matrix_calloc(1, 1);

			pipeline[i].type = 8;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "SIGM") == 0)
		{
			SIGM* aux = (SIGM*) malloc(sizeof(SIGM));

			aux->batch_size = batch_size;
			aux->n_units = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_units"))[0];

			aux->a = gsl_matrix_calloc(aux->batch_size, aux->n_units);

			pipeline[i].type = 9;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "DIRE") == 0)
		{
			DIRE* aux = (DIRE*) malloc(sizeof(DIRE));

			aux->batch_size = batch_size;
			aux->n_units = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_units"))[0];

			aux->buff_x = gsl_matrix_calloc(aux->batch_size, aux->n_units);
			aux->buff_dy = gsl_matrix_calloc(aux->batch_size, aux->n_units);

			pipeline[i].type = 11;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "TANH") == 0)
		{
			TANH* aux = (TANH*) malloc(sizeof(TANH));

			aux->batch_size = batch_size;
			aux->n_units = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_units"))[0];

			aux->a = gsl_matrix_calloc(aux->batch_size, aux->n_units);

			pipeline[i].type = 12;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "RBML") == 0)
		{
			RBML* aux = (RBML*) malloc(sizeof(RBML));
			
			aux->batch_size = batch_size;
			aux->n_visible = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_visible"))[0];
			aux->n_hidden = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_hidden"))[0];
			aux->n_gibbs = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_gibbs"))[0];

			SEXP W = PROTECT(getListElement(VECTOR_ELT(layers, i), "W"));
			SEXP grad_W = PROTECT(getListElement(VECTOR_ELT(layers, i), "grad_W"));
			SEXP vbias = PROTECT(getListElement(VECTOR_ELT(layers, i), "vbias"));
			SEXP grad_vbias = PROTECT(getListElement(VECTOR_ELT(layers, i), "grad_vbias"));
			SEXP hbias = PROTECT(getListElement(VECTOR_ELT(layers, i), "hbias"));
			SEXP grad_hbias = PROTECT(getListElement(VECTOR_ELT(layers, i), "grad_hbias"));

			aux->W = gsl_matrix_calloc(aux->n_hidden, aux->n_visible);
			aux->grad_W = gsl_matrix_calloc(aux->n_hidden, aux->n_visible);
			aux->vbias = gsl_vector_calloc(aux->n_visible);
			aux->grad_vbias = gsl_vector_calloc(aux->n_visible);
			aux->hbias = gsl_vector_calloc(aux->n_hidden);
			aux->grad_hbias = gsl_vector_calloc(aux->n_hidden);

			for (int h = 0; h < aux->n_hidden; h++)
			{
				for (int v = 0; v < aux->n_visible; v++)
				{
					gsl_matrix_set(aux->W, h, v, RMATRIX(W,h,v));
					gsl_matrix_set(aux->grad_W, h, v, RMATRIX(grad_W,h,v));
				}
				gsl_vector_set(aux->hbias, h, RVECTOR(hbias, h));
				gsl_vector_set(aux->grad_hbias, h, RVECTOR(grad_hbias, h));

			}
			for (int v = 0; v < aux->n_visible; v++)
			{
				gsl_vector_set(aux->vbias, v, RVECTOR(vbias, v));
				gsl_vector_set(aux->grad_vbias, v, RVECTOR(grad_vbias, v));
			}
			UNPROTECT(6);
			
			aux->x = gsl_matrix_calloc(aux->batch_size, aux->n_visible);
			aux->ph_means = gsl_matrix_calloc(aux->batch_size, aux->n_hidden);

			pipeline[i].type = 13;
			pipeline[i].layer = (void*) aux;
		}
		else // XENT, MSEL, others...
		{
			pipeline[i].type = 0;
			pipeline[i].layer = NULL; // Let the system explode, for now...
		}
	}

	return pipeline;
}

LAYER* build_loss_layer(SEXP eval_layer, int batch_size)
{
	LAYER* retval = (LAYER*) malloc(sizeof(LAYER));
	if (strcmp(CHAR(STRING_ELT(eval_layer, 0)), "XENT") == 0)
	{
		XENT* xent = (XENT*) malloc(sizeof(XENT));
		create_XENT(xent);
		retval->type = 7; retval->layer = (void*) xent;
	}
	else if (strcmp(CHAR(STRING_ELT(eval_layer, 0)), "RBML") == 0)
	{
		RBML* rbml = (RBML*) malloc(sizeof(RBML));

		int nvis_rbml = atoi(CHAR(STRING_ELT(eval_layer, 1)));     // n_visible
		int nhid_rbml = atoi(CHAR(STRING_ELT(eval_layer, 2)));     // n_hidden
		double scale_rbml = atof(CHAR(STRING_ELT(eval_layer, 3))); // scale
		int n_gibbs_rbml = atof(CHAR(STRING_ELT(eval_layer, 4)));  // n_gibbs

		create_RBML(rbml, nvis_rbml, nhid_rbml, scale_rbml, n_gibbs_rbml, batch_size);
		retval->type = 13; retval->layer = (void*) rbml;
	}

	return retval;
}

void free_loss_layer (LAYER* layer)
{
	if (layer->type == 7) free_XENT((XENT*) layer->layer);
	else if (layer->type == 13) free_RBML((RBML*) layer->layer);
	free(layer->layer);
	free(layer);
}

/*----------------------------------------------------------------------------*/
/* TRAIN/PREDICT FUNCTIONS FOR CNN                                            */ 
/*----------------------------------------------------------------------------*/

// Interface for Training a CNN
SEXP _C_CNN_train (SEXP dataset, SEXP targets, SEXP layers, SEXP num_layers, SEXP eval_layer, SEXP batch_size,
	SEXP training_epochs, SEXP learning_rate, SEXP momentum, SEXP rand_seed, SEXP is_init_cnn, SEXP is_dbn)
{
 	int nrows = INTEGER(GET_DIM(dataset))[0];
 	int nchan = INTEGER(GET_DIM(dataset))[1];
 	int img_h = INTEGER(GET_DIM(dataset))[2];
 	int img_w = INTEGER(GET_DIM(dataset))[3];

 	int nouts = INTEGER(GET_DIM(targets))[1];

 	int basi = INTEGER_VALUE(batch_size);
 	int trep = INTEGER_VALUE(training_epochs);
 	double lera = NUMERIC_VALUE(learning_rate);
 	double mome = NUMERIC_VALUE(momentum);

	int nlays = INTEGER_VALUE(num_layers);
	
	int rebuild = INTEGER_VALUE(is_init_cnn);
	int no_cmat = INTEGER_VALUE(is_dbn);

 	unsigned int rase = INTEGER_VALUE(rand_seed);
	srand(rase);

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

	// Build the Layers pipeline or re-assemble an initial CNN
	LAYER* pipeline;
	if (rebuild == 0) pipeline = build_pipeline(layers, nlays, basi);
	else pipeline = reassemble_CNN(layers, nlays, basi);
	
	// Build the Evaluation Layer
	LAYER* loss_layer = build_loss_layer(eval_layer, basi);

	// Train a CNN
	double loss = train_cnn (train_X, train_Y, nrows, nchan, pipeline, nlays, loss_layer, trep, basi, lera, mome, rase);

	// Pass the Training set through the CNN
	gsl_matrix* predictions = prediction_cnn (train_X, nrows, nchan, pipeline, nlays, basi);
	gsl_matrix* confusion;
	if (no_cmat == 0) confusion = classification_matrix(predictions, train_Y);
	else confusion = gsl_matrix_calloc(nouts,nouts);

	// Return Structure
	SEXP retval = PROTECT(allocVector(VECSXP, 7));

	SET_VECTOR_ELT(retval, 0, allocVector(INTSXP, 4));
	INTEGER(VECTOR_ELT(retval, 0))[0] = nrows;
	INTEGER(VECTOR_ELT(retval, 0))[1] = nchan;
	INTEGER(VECTOR_ELT(retval, 0))[2] = img_h;
	INTEGER(VECTOR_ELT(retval, 0))[3] = img_w;

	SET_VECTOR_ELT(retval, 1, allocVector(INTSXP, 2));
	INTEGER(VECTOR_ELT(retval, 1))[0] = nrows;
	INTEGER(VECTOR_ELT(retval, 1))[1] = nouts;

	SET_VECTOR_ELT(retval, 2, allocVector(VECSXP, nlays));
	return_pipeline (&retval, pipeline, nlays);

	SET_VECTOR_ELT(retval, 3, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 3))[0] = nlays;

	SET_VECTOR_ELT(retval, 4, allocMatrix(REALSXP, nouts, nouts));
	for (int i = 0; i < nouts; i++)
		for (int j = 0; j < nouts; j++)
			REAL(VECTOR_ELT(retval, 4))[j * nouts + i] = gsl_matrix_get(confusion, i, j);

	SET_VECTOR_ELT(retval, 5, allocVector(REALSXP, 1));
	REAL(VECTOR_ELT(retval, 5))[0] = loss;

	SET_VECTOR_ELT(retval, 6, allocMatrix(REALSXP, nrows, nouts));
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < nouts; j++)
			REAL(VECTOR_ELT(retval, 6))[j * nrows + i] = gsl_matrix_get(predictions, i, j);

	SEXP nms = PROTECT(allocVector(STRSXP, 7));
	SET_STRING_ELT(nms, 0, mkChar("dims.in"));
	SET_STRING_ELT(nms, 1, mkChar("dims.out"));
	SET_STRING_ELT(nms, 2, mkChar("layers"));
	SET_STRING_ELT(nms, 3, mkChar("n.layers"));
	SET_STRING_ELT(nms, 4, mkChar("conf.matrix"));
	SET_STRING_ELT(nms, 5, mkChar("mean.loss"));
	SET_STRING_ELT(nms, 6, mkChar("fitted.values"));
	setAttrib(retval, R_NamesSymbol, nms);

	UNPROTECT(2 + nlays);

	// Free basic structures
	free_pipeline(pipeline, nlays);
	free_loss_layer(loss_layer);

	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < nchan; j++)
			gsl_matrix_free(train_X[i][j]);
		free(train_X[i]);
	}
	free(train_X);
	gsl_matrix_free(train_Y);
	gsl_matrix_free(predictions);
	gsl_matrix_free(confusion);

	return retval;
}

// Interface for Predicting using a CNN
SEXP _C_CNN_predict (SEXP newdata, SEXP layers, SEXP num_layers, SEXP rand_seed)
{
 	int nrows = INTEGER(GET_DIM(newdata))[0];
 	int nchan = INTEGER(GET_DIM(newdata))[1];
 	int img_h = INTEGER(GET_DIM(newdata))[2];
 	int img_w = INTEGER(GET_DIM(newdata))[3];

 	int nlay = INTEGER_VALUE(num_layers);

	int basi = min(100, nrows);
	
	unsigned int rase = INTEGER_VALUE(rand_seed);
	srand(rase);

	// Re-assemble the CNN (build pipeline)
	LAYER* pipeline = reassemble_CNN(layers, nlay, basi);

	// Prepare Test Dataset
	gsl_matrix*** test_X = (gsl_matrix***) malloc(nrows * sizeof(gsl_matrix**));
	for (int r = 0; r < nrows; r++)
	{
		test_X[r] = (gsl_matrix**) malloc(nchan * sizeof(gsl_matrix*));
		for (int c = 0; c < nchan; c++)
		{
			test_X[r][c] = gsl_matrix_alloc(img_h, img_w);
			for (int h = 0; h < img_h; h++)
				for (int w = 0; w < img_w; w++)
					gsl_matrix_set(test_X[r][c], h, w, RARRAY(newdata, r, c, h, w));
		}
	}

	// Pass through CNN
	gsl_matrix* predictions = prediction_cnn (test_X, nrows, nchan, pipeline, nlay, basi);
	int nouts = (int) predictions->size2;

	// Prepare Results
	SEXP retval = PROTECT(allocMatrix(REALSXP, nrows, nouts));
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < nouts; j++)
			REAL(retval)[j * nrows + i] = gsl_matrix_get(predictions, i, j);

	// Free the structures and the CNN
	free_pipeline (pipeline, nlay);

	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < nchan; j++)
			gsl_matrix_free(test_X[i][j]);
		free(test_X[i]);
	}
	free(test_X);
	gsl_matrix_free(predictions);

	UNPROTECT(1);

	return retval;
}

// Interface for Predicting and Reconstructing using a CNN
SEXP _C_CNN_pass_through (SEXP newdata, SEXP layers, SEXP num_layers, SEXP rand_seed)
{
 	int nrows = INTEGER(GET_DIM(newdata))[0];
 	int nchan = INTEGER(GET_DIM(newdata))[1];
 	int img_h = INTEGER(GET_DIM(newdata))[2];
 	int img_w = INTEGER(GET_DIM(newdata))[3];

 	int nlay = INTEGER_VALUE(num_layers);

	int basi = min(100, nrows);
	
	unsigned int rase = INTEGER_VALUE(rand_seed);
	srand(rase);

	// Re-assemble the CNN (build pipeline)
	LAYER* pipeline = reassemble_CNN(layers, nlay, basi);

	// Prepare Test Dataset
	gsl_matrix*** test_X = (gsl_matrix***) malloc(nrows * sizeof(gsl_matrix**));
	for (int r = 0; r < nrows; r++)
	{
		test_X[r] = (gsl_matrix**) malloc(nchan * sizeof(gsl_matrix*));
		for (int c = 0; c < nchan; c++)
		{
			test_X[r][c] = gsl_matrix_alloc(img_h, img_w);
			for (int h = 0; h < img_h; h++)
				for (int w = 0; w < img_w; w++)
					gsl_matrix_set(test_X[r][c], h, w, RARRAY(newdata, r, c, h, w));
		}
	}

	// Pass through CNN
	gsl_matrix* features = NULL;
	gsl_matrix*** rebuild = NULL;
	pass_through_cnn (test_X, nrows, nchan, pipeline, nlay, basi, &features, &rebuild);
	int nouts = (int) features->size2;

	// Prepare Results
	SEXP retval = PROTECT(allocVector(VECSXP, 2));

	SET_VECTOR_ELT(retval, 0, allocMatrix(REALSXP, nrows, nouts));
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < nouts; j++)
			REAL(VECTOR_ELT(retval, 0))[j * nrows + i] = gsl_matrix_get(features, i, j);
	
	SEXP dim;
	PROTECT(dim = Rf_allocVector(INTSXP, 4));
	INTEGER(dim)[0] = nrows;
	INTEGER(dim)[1] = nchan;
	INTEGER(dim)[2] = img_h;
	INTEGER(dim)[3] = img_w;

	SET_VECTOR_ELT(retval, 1, allocArray(REALSXP, dim));
	for (int b = 0; b < nrows; b++)
		for (int c = 0; c < nchan; c++)
			for (int h = 0; h < img_h; h++)
				for (int w = 0; w < img_w; w++)
				{
					int idx = w + img_w * (h + img_h * (c + nchan * b));
					REAL(VECTOR_ELT(retval, 1))[idx] = gsl_matrix_get(rebuild[b][c], h, w);
				}

	SEXP nms = PROTECT(allocVector(STRSXP, 2));
	SET_STRING_ELT(nms, 0, mkChar("features"));
	SET_STRING_ELT(nms, 1, mkChar("rebuild"));
	setAttrib(retval, R_NamesSymbol, nms);

	// Free the structures and the CNN
	free_pipeline (pipeline, nlay);

	for (int i = 0; i < nrows; i++)
	{
		for (int j = 0; j < nchan; j++)
		{
			gsl_matrix_free(test_X[i][j]);
			gsl_matrix_free(rebuild[i][j]);
		}
		free(test_X[i]);
		free(rebuild[i]);
	}
	free(test_X);
	free(rebuild);
	gsl_matrix_free(features);
	
	UNPROTECT(3);

	return retval;
}

/*----------------------------------------------------------------------------*/
/* TRAIN/PREDICT FUNCTIONS FOR MLP                                            */
/*----------------------------------------------------------------------------*/

// Interface for Training a MLP
SEXP _C_MLP_train (SEXP dataset, SEXP targets, SEXP layers, SEXP num_layers, SEXP eval_layer, SEXP batch_size,
	SEXP training_epochs, SEXP learning_rate, SEXP momentum, SEXP rand_seed, SEXP is_init_cnn, SEXP is_dbn)
{
 	int nrows = INTEGER(GET_DIM(dataset))[0];
 	int ncols = INTEGER(GET_DIM(dataset))[1];

 	int nouts = INTEGER(GET_DIM(targets))[1];

 	int basi = INTEGER_VALUE(batch_size);
 	int trep = INTEGER_VALUE(training_epochs);
 	double lera = NUMERIC_VALUE(learning_rate);
 	double mome = NUMERIC_VALUE(momentum);

	int nlays = INTEGER_VALUE(num_layers);
	
	int rebuild = INTEGER_VALUE(is_init_cnn);
	int no_cmat = INTEGER_VALUE(is_dbn);

 	unsigned int rase = INTEGER_VALUE(rand_seed);
	srand(rase);
	
	// Create Dataset Structure
	gsl_matrix* train_X = gsl_matrix_alloc(nrows, ncols);
	gsl_matrix* train_Y = gsl_matrix_alloc(nrows, nouts);
	for (int r = 0; r < nrows; r++)
	{
		for (int c = 0; c < ncols; c++)
			gsl_matrix_set(train_X, r, c, RMATRIX(dataset, r, c));

		for (int o = 0; o < nouts; o++)
			gsl_matrix_set(train_Y, r, o, RMATRIX(targets, r, o));
	}

	// Build the Layers pipeline or re-assemble an initial CNN
	LAYER* pipeline;
	if (rebuild == 0) pipeline = build_pipeline(layers, nlays, basi);
	else pipeline = reassemble_CNN(layers, nlays, basi);

	// Build the Evaluation Layer
	LAYER* loss_layer = build_loss_layer(eval_layer, basi);

	// Train a MLP
	double loss = train_mlp(train_X, train_Y, pipeline, nlays, loss_layer, trep, basi, lera, mome, rase);

	// Pass the Training set through the MLP
	gsl_matrix* predictions = prediction_mlp(train_X, pipeline, nlays, basi);
	gsl_matrix* confusion;
	if (no_cmat == 0) confusion = classification_matrix(predictions, train_Y);
	else confusion = gsl_matrix_calloc(nouts,nouts);

	// Return Structure
	SEXP retval = PROTECT(allocVector(VECSXP, 7));

	SET_VECTOR_ELT(retval, 0, allocVector(INTSXP, 2));
	INTEGER(VECTOR_ELT(retval, 0))[0] = nrows;
	INTEGER(VECTOR_ELT(retval, 0))[1] = ncols;

	SET_VECTOR_ELT(retval, 1, allocVector(INTSXP, 2));
	INTEGER(VECTOR_ELT(retval, 1))[0] = nrows;
	INTEGER(VECTOR_ELT(retval, 1))[1] = nouts;

	SET_VECTOR_ELT(retval, 2, allocVector(VECSXP, nlays));
	return_pipeline (&retval, pipeline, nlays);

	SET_VECTOR_ELT(retval, 3, allocVector(INTSXP, 1));
	INTEGER(VECTOR_ELT(retval, 3))[0] = nlays;

	SET_VECTOR_ELT(retval, 4, allocMatrix(REALSXP, nouts, nouts));
	for (int i = 0; i < nouts; i++)
		for (int j = 0; j < nouts; j++)
			REAL(VECTOR_ELT(retval, 4))[j * nouts + i] = gsl_matrix_get(confusion, i, j); //CHECK!

	SET_VECTOR_ELT(retval, 5, allocVector(REALSXP, 1));
	REAL(VECTOR_ELT(retval, 5))[0] = loss;

	SET_VECTOR_ELT(retval, 6, allocMatrix(REALSXP, nrows, nouts));
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < nouts; j++)
			REAL(VECTOR_ELT(retval, 6))[j * nrows + i] = gsl_matrix_get(predictions, i, j); //CHECK!

	SEXP nms = PROTECT(allocVector(STRSXP, 7));
	SET_STRING_ELT(nms, 0, mkChar("dims.in"));
	SET_STRING_ELT(nms, 1, mkChar("dims.out"));
	SET_STRING_ELT(nms, 2, mkChar("layers"));
	SET_STRING_ELT(nms, 3, mkChar("n.layers"));
	SET_STRING_ELT(nms, 4, mkChar("conf.matrix"));
	SET_STRING_ELT(nms, 5, mkChar("mean.loss"));
	SET_STRING_ELT(nms, 6, mkChar("fitted.values"));
	setAttrib(retval, R_NamesSymbol, nms);

	UNPROTECT(2 + nlays);

	// Free basic structures
	free_pipeline (pipeline, nlays);
	free_loss_layer(loss_layer);

	gsl_matrix_free(train_X);
	gsl_matrix_free(train_Y);
	gsl_matrix_free(predictions);
	gsl_matrix_free(confusion);

	return retval;
}

// Interface for Predicting using a MLP
SEXP _C_MLP_predict (SEXP newdata, SEXP layers, SEXP num_layers, SEXP rand_seed)
{
 	int nrows = INTEGER(GET_DIM(newdata))[0];
 	int ncols = INTEGER(GET_DIM(newdata))[1];

 	int nlay = INTEGER_VALUE(num_layers);
 	
	int basi = min(100, nrows);
	
	unsigned int rase = INTEGER_VALUE(rand_seed);
	srand(rase);

	// Re-assemble the MLP (build pipeline)
	LAYER* pipeline = reassemble_CNN(layers, nlay, basi);

	// Prepare Test Dataset
	gsl_matrix* test_X = gsl_matrix_alloc(nrows, ncols);
	for (int r = 0; r < nrows; r++)
		for (int c = 0; c < ncols; c++)
			gsl_matrix_set(test_X, r, c, RMATRIX(newdata, r, c));

	// Pass through MLP
	gsl_matrix* predictions = prediction_mlp (test_X, pipeline, nlay, basi);
	int nouts = (int) predictions->size2;

	// Prepare Results
	SEXP retval = PROTECT(allocMatrix(REALSXP, nrows, nouts));
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < nouts; j++)
			REAL(retval)[j * nrows + i] = gsl_matrix_get(predictions, i, j); //CHECK!

	// Free the structures and the MLP
	free_pipeline (pipeline, nlay);

	gsl_matrix_free(test_X);
	gsl_matrix_free(predictions);

	UNPROTECT(1);

	return retval;
}

// Interface for Predicting and Reconstructing using a MLP
SEXP _C_MLP_pass_through (SEXP newdata, SEXP layers, SEXP num_layers, SEXP rand_seed)
{
 	int nrows = INTEGER(GET_DIM(newdata))[0];
 	int ncols = INTEGER(GET_DIM(newdata))[1];

 	int nlay = INTEGER_VALUE(num_layers);
 	
	int basi = min(100, nrows);
	
	unsigned int rase = INTEGER_VALUE(rand_seed);
	srand(rase);

	// Re-assemble the MLP (build pipeline)
	LAYER* pipeline = reassemble_CNN(layers, nlay, basi);

	// Prepare Test Dataset
	gsl_matrix* test_X = gsl_matrix_alloc(nrows, ncols);
	for (int r = 0; r < nrows; r++)
		for (int c = 0; c < ncols; c++)
			gsl_matrix_set(test_X, r, c, RMATRIX(newdata, r, c));

	// Pass through MLP
	gsl_matrix* features = NULL;
	gsl_matrix* rebuild = NULL;
	pass_through_mlp (test_X, pipeline, nlay, basi, &features, &rebuild);
	int nouts = (int) features->size2;

	// Prepare Results
	SEXP retval = PROTECT(allocVector(VECSXP, 2));

	SET_VECTOR_ELT(retval, 0, allocMatrix(REALSXP, nrows, nouts));
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < nouts; j++)
			REAL(VECTOR_ELT(retval, 0))[j * nrows + i] = gsl_matrix_get(features, i, j);

	SET_VECTOR_ELT(retval, 1, allocMatrix(REALSXP, nrows, ncols));
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncols; j++)
			REAL(VECTOR_ELT(retval, 1))[j * nrows + i] = gsl_matrix_get(rebuild, i, j);
	
	SEXP nms = PROTECT(allocVector(STRSXP, 2));
	SET_STRING_ELT(nms, 0, mkChar("features"));
	SET_STRING_ELT(nms, 1, mkChar("rebuild"));
	setAttrib(retval, R_NamesSymbol, nms);

	// Free the structures and the MLP
	free_pipeline (pipeline, nlay);

	gsl_matrix_free(test_X);
	gsl_matrix_free(features);
	gsl_matrix_free(rebuild);

	UNPROTECT(2);

	return retval;
}

