/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// File including R interface for CNNs
// Compile using "R CMD SHLIB cnn_r.c cell.c dire.c flat.c line.c matrix_ops.c msel.c relu.c sigm.c test.c cnn.c conv.c grad_check.c mlp.c pool.c relv.c soft.c -lgsl -lgslcblas -lm -o cnn.so"

#include "cnn.h"

#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>

/*---------------------------------------------------------------------------*/
/* INTERFACE TO R                                                            */
/*---------------------------------------------------------------------------*/

#define RMATRIX(m,i,j) (REAL(m)[ INTEGER(GET_DIM(m))[0]*(j)+(i) ])
#define RVECTOR(v,i) (REAL(v)[(i)])
#define RVECTORI(v,i) (INTEGER(v)[(i)])

LAYER* build_pipeline (SEXP layers, int nlays)
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
			int bsize_conv = atoi(CHAR(STRING_ELT(laux, 6))); // batch_size

			create_CONV(conv, nchan_conv, nfilt_conv, fsize_conv, scale_conv, bmode_conv, bsize_conv);
			retval[i].type = 1; retval[i].layer = (void*) conv;
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
			retval[i].type = 2; retval[i].layer = (void*) pool;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "RELU") == 0)
		{
			RELU* relu = (RELU*) malloc(sizeof(RELU));

			int nchan_relu = atoi(CHAR(STRING_ELT(laux, 1))); // n_channels
			int bsize_relu = atoi(CHAR(STRING_ELT(laux, 2))); // batch_size

			create_RELU(relu, nchan_relu, bsize_relu);
			retval[i].type = 3; retval[i].layer = (void*) relu;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "FLAT") == 0)
		{
			FLAT* flat = (FLAT*) malloc(sizeof(FLAT));

			int nchan_flat = atoi(CHAR(STRING_ELT(laux, 1))); // n_channels
			int bsize_flat = atoi(CHAR(STRING_ELT(laux, 2))); // batch_size

			create_FLAT(flat, nchan_flat, bsize_flat);
			retval[i].type = 4; retval[i].layer = (void*) flat;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "LINE") == 0)
		{
			LINE* line = (LINE*) malloc(sizeof(LINE));

			int nvis_line = atoi(CHAR(STRING_ELT(laux, 1))); // n_visible
			int nhid_line = atoi(CHAR(STRING_ELT(laux, 2))); // n_visible
			double scale_line = atof(CHAR(STRING_ELT(laux, 3))); // scale
			int bsize_line = atoi(CHAR(STRING_ELT(laux, 4))); // batch_size

			create_LINE(line, nvis_line, nhid_line, scale_line, bsize_line);
			retval[i].type = 5; retval[i].layer = (void*) line;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "SOFT") == 0)
		{
			SOFT* soft = (SOFT*) malloc(sizeof(SOFT));

			int nunits_soft = atoi(CHAR(STRING_ELT(laux, 1))); // n_units
			int bsize_soft = atoi(CHAR(STRING_ELT(laux, 2))); // batch_size

			create_SOFT(soft, nunits_soft, bsize_soft);
			retval[i].type = 6; retval[i].layer = (void*) soft;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "RELV") == 0)
		{
			RELV* relv = (RELV*) malloc(sizeof(RELV));

			int bsize_relv = atoi(CHAR(STRING_ELT(laux, 1))); // batch_size

			create_RELV(relv, bsize_relv);
			retval[i].type = 8; retval[i].layer = (void*) relv;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "SIGM") == 0)
		{
			SIGM* sigm = (SIGM*) malloc(sizeof(SIGM));

			int nunits_sigm = atoi(CHAR(STRING_ELT(laux, 1))); // n_units
			int bsize_sigm = atoi(CHAR(STRING_ELT(laux, 2))); // batch_size

			create_SIGM(sigm, nunits_sigm, bsize_sigm);
			retval[i].type = 9; retval[i].layer = (void*) sigm;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "DIRE") == 0)
		{
			DIRE* dire = (DIRE*) malloc(sizeof(DIRE));

			int nunits_dire = atoi(CHAR(STRING_ELT(laux, 1))); // n_units
			int bsize_dire = atoi(CHAR(STRING_ELT(laux, 2))); // batch_size

			create_DIRE(dire, nunits_dire, bsize_dire);
			retval[i].type = 11; retval[i].layer = (void*) dire;
		}
		else if (strcmp(CHAR(STRING_ELT(laux, 0)), "TANH") == 0)
		{
			TANH* tanh = (TANH*) malloc(sizeof(TANH));

			int nunits_tanh = atoi(CHAR(STRING_ELT(laux, 1))); // n_units
			int bsize_tanh = atoi(CHAR(STRING_ELT(laux, 2))); // batch_size

			create_TANH(tanh, nunits_tanh, bsize_tanh);
			retval[i].type = 12; retval[i].layer = (void*) tanh;
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
	}
	return;
}

void return_pipeline (SEXP* retval, LAYER* pipeline, int nlays)
{
	for (int i = 0; i < nlays; i++)
	{
		if (pipeline[i].type == 1) // CONV Layer
		{
			CONV* aux = (CONV*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 11));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("CONV"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->batch_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->filter_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3))[0] = aux->n_filters;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4))[0] = aux->n_channels;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5, allocVector(INTSXP, 2));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5))[0] = aux->pad_y;
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5))[1] = aux->pad_x;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 6, allocVector(INTSXP, 2));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 6))[0] = aux->win_h;
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 6))[1] = aux->win_w;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 7, allocVector(REALSXP, aux->n_filters));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 8, allocVector(REALSXP, aux->n_filters));
			for (int j = 0; j < aux->n_filters; j++)
			{
				REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 7))[j] = gsl_vector_get(aux->b, j);
				REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 8))[j] = gsl_vector_get(aux->grad_b, j);
			}

			SEXP dim;
			PROTECT(dim = Rf_allocVector(INTSXP, 4));
			INTEGER(dim)[0] = aux->n_filters;
			INTEGER(dim)[1] = aux->n_channels;
			INTEGER(dim)[2] = aux->filter_size;
			INTEGER(dim)[3] = aux->filter_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 9, allocArray(REALSXP, dim));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 10, allocArray(REALSXP, dim));
			for (int f = 0; f < aux->n_filters; f++)
				for (int c = 0; c < aux->n_channels; c++)
					for (int h = 0; h < aux->filter_size; h++)
						for (int w = 0; w < aux->filter_size; w++)
						{
							int idx = w + aux->filter_size * (h + aux->filter_size * (c + aux->n_channels * f));
							REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 9))[idx] = gsl_matrix_get(aux->W[f][c], h, w);
							REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 10))[idx] = gsl_matrix_get(aux->grad_W[f][c], h, w);
						}

			UNPROTECT(1);

			SEXP naux = PROTECT(allocVector(STRSXP, 11));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("batch_size"));
			SET_STRING_ELT(naux, 2, mkChar("filter_size"));
			SET_STRING_ELT(naux, 3, mkChar("n_filters"));
			SET_STRING_ELT(naux, 4, mkChar("n_channels"));
			SET_STRING_ELT(naux, 5, mkChar("padding"));
			SET_STRING_ELT(naux, 6, mkChar("win_size"));
			SET_STRING_ELT(naux, 7, mkChar("b"));
			SET_STRING_ELT(naux, 8, mkChar("grad_b"));
			SET_STRING_ELT(naux, 9, mkChar("W"));
			SET_STRING_ELT(naux, 10, mkChar("grad_W"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);
		}
		else if (pipeline[i].type == 2) // POOL Layer
		{
			POOL* aux = (POOL*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 6));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("POOL"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->batch_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->n_channels;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3))[0] = aux->win_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4))[0] = aux->stride;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5))[0] = aux->padding;

			SEXP naux = PROTECT(allocVector(STRSXP, 6));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("batch_size"));
			SET_STRING_ELT(naux, 2, mkChar("n_channels"));
			SET_STRING_ELT(naux, 3, mkChar("win_size"));
			SET_STRING_ELT(naux, 4, mkChar("stride"));
			SET_STRING_ELT(naux, 5, mkChar("padding"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 3) // RELU Layer
		{
			RELU* aux = (RELU*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 3));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("RELU"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->batch_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->n_channels;

			SEXP naux = PROTECT(allocVector(STRSXP, 3));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("batch_size"));
			SET_STRING_ELT(naux, 2, mkChar("n_channels"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 4) // FLAT Layer
		{
			FLAT* aux = (FLAT*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 4));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("FLAT"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->batch_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->n_channels;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3, allocVector(INTSXP, 2));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3))[0] = aux->img_h;
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3))[1] = aux->img_w;

			SEXP naux = PROTECT(allocVector(STRSXP, 4));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("batch_size"));
			SET_STRING_ELT(naux, 2, mkChar("n_channels"));
			SET_STRING_ELT(naux, 3, mkChar("img_dims"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 5) // LINE Layer
		{
			LINE* aux = (LINE*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 8));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("LINE"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->batch_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->n_visible;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3))[0] = aux->n_hidden;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4, allocVector(REALSXP, aux->n_hidden));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5, allocVector(REALSXP, aux->n_hidden));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 6, allocMatrix(REALSXP, aux->n_hidden, aux->n_visible));
			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 7, allocMatrix(REALSXP, aux->n_hidden, aux->n_visible));
			for (int j = 0; j < aux->n_hidden; j++)
			{
				for (int k = 0; k < aux->n_visible; k++)
				{
					REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 6))[j * aux->n_visible + k] = gsl_matrix_get(aux->W, j, k);
					REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 7))[j * aux->n_visible + k] = gsl_matrix_get(aux->grad_W, j, k);
				}
				REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4))[j] = gsl_vector_get(aux->b, j);
				REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 5))[j] = gsl_vector_get(aux->grad_b, j);
			}

			SEXP naux = PROTECT(allocVector(STRSXP, 8));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("batch_size"));
			SET_STRING_ELT(naux, 2, mkChar("n_visible"));
			SET_STRING_ELT(naux, 3, mkChar("n_hidden"));
			SET_STRING_ELT(naux, 4, mkChar("b"));
			SET_STRING_ELT(naux, 5, mkChar("grad_b"));
			SET_STRING_ELT(naux, 6, mkChar("W"));
			SET_STRING_ELT(naux, 7, mkChar("grad_W"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 6) // SOFT Layer
		{
			SOFT* aux = (SOFT*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 3));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("SOFT"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->batch_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->n_units;

			SEXP naux = PROTECT(allocVector(STRSXP, 3));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("batch_size"));
			SET_STRING_ELT(naux, 2, mkChar("n_units"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 8) // RELV Layer
		{
			RELV* aux = (RELV*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 2));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("RELV"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->batch_size;

			SEXP naux = PROTECT(allocVector(STRSXP, 2));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("batch_size"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else if (pipeline[i].type == 9) // SIGM Layer
		{
			SIGM* aux = (SIGM*) pipeline[i].layer;

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 3));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("SIGM"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->batch_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->n_units;

			SEXP naux = PROTECT(allocVector(STRSXP, 3));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("batch_size"));
			SET_STRING_ELT(naux, 2, mkChar("n_units"));

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
					REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 3))[j * aux->n_units + k] = gsl_matrix_get(aux->buff_x, j, k);
					REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 4))[j * aux->n_units + k] = gsl_matrix_get(aux->buff_dy, j, k);
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

			SET_VECTOR_ELT(VECTOR_ELT(*retval, 2), i, allocVector(VECSXP, 3));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 0, allocVector(STRSXP, 1));
			SET_STRING_ELT(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i),0), 0, mkChar("TANH"));

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 1))[0] = aux->batch_size;

			SET_VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2, allocVector(INTSXP, 1));
			INTEGER(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), 2))[0] = aux->n_units;

			SEXP naux = PROTECT(allocVector(STRSXP, 3));
			SET_STRING_ELT(naux, 0, mkChar("type"));
			SET_STRING_ELT(naux, 1, mkChar("batch_size"));
			SET_STRING_ELT(naux, 2, mkChar("n_units"));

			setAttrib(VECTOR_ELT(VECTOR_ELT(*retval, 2), i), R_NamesSymbol, naux);	
		}
		else // CELL, MSEL, others...
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

#define RARRAY(m,i,j,k,l) (REAL(m)[INTEGER(GET_DIM(m))[2] * INTEGER(GET_DIM(m))[1] * INTEGER(GET_DIM(m))[0] * (l) + INTEGER(GET_DIM(m))[1] * INTEGER(GET_DIM(m))[0] * (k) + INTEGER(GET_DIM(m))[0] * (j) + (i) ])

// Interface for Training a CNN
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

	// Build the Layers pipeline
	LAYER* pipeline = build_pipeline(layers, nlays);

	// Train a CNN
	double loss = train_cnn (train_X, train_Y, nrows, nchan, pipeline, nlays, trep, basi, lera, mome, rase);

	// Pass the Training set through the CNN
	gsl_matrix* predictions = prediction_cnn (train_X, nrows, nchan, pipeline, nlays);
	gsl_matrix* confusion = classification_matrix(predictions, train_Y);

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
			REAL(VECTOR_ELT(retval, 4))[i * nouts + j] = gsl_matrix_get(confusion, i, j);

	SET_VECTOR_ELT(retval, 5, allocVector(REALSXP, 1));
	REAL(VECTOR_ELT(retval, 5))[0] = loss;

	SET_VECTOR_ELT(retval, 6, allocMatrix(REALSXP, nrows, nouts));
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < nouts; j++)
			REAL(VECTOR_ELT(retval, 6))[i * nouts + j] = gsl_matrix_get(predictions, i, j);

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

// Function to Re-assemble the CNN
LAYER* reassemble_CNN (SEXP layers, int num_layers)
{
	LAYER* pipeline = (LAYER*) malloc(num_layers * sizeof(LAYER));

	for (int i = 0; i < num_layers; i++)
	{
		const char* s = CHAR(STRING_ELT(getListElement(VECTOR_ELT(layers, i), "type"),0));
		if (strcmp(s, "CONV") == 0)
		{
			CONV* aux = (CONV*) malloc(sizeof(CONV));

			aux->batch_size = INTEGER(getListElement(VECTOR_ELT(layers, i), "batch_size"))[0];
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

			aux->batch_size = INTEGER(getListElement(VECTOR_ELT(layers, i), "batch_size"))[0];
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

			aux->batch_size = INTEGER(getListElement(VECTOR_ELT(layers, i), "batch_size"))[0];
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

			aux->batch_size = INTEGER(getListElement(VECTOR_ELT(layers, i), "batch_size"))[0];
			aux->n_channels = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_channels"))[0];
			aux->img_h = INTEGER(getListElement(VECTOR_ELT(layers, i), "img_dims"))[0];
			aux->img_w = INTEGER(getListElement(VECTOR_ELT(layers, i), "img_dims"))[1];

			pipeline[i].type = 4;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "LINE") == 0)
		{
			LINE* aux = (LINE*) malloc(sizeof(LINE));

			aux->batch_size = INTEGER(getListElement(VECTOR_ELT(layers, i), "batch_size"))[0];
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

			aux->batch_size = INTEGER(getListElement(VECTOR_ELT(layers, i), "batch_size"))[0];
			aux->n_units = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_units"))[0];

			pipeline[i].type = 6;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "RELV") == 0)
		{
			RELV* aux = (RELV*) malloc(sizeof(RELV));

			aux->batch_size = INTEGER(getListElement(VECTOR_ELT(layers, i), "batch_size"))[0];

			aux->img = gsl_matrix_calloc(1, 1);

			pipeline[i].type = 8;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "SIGM") == 0)
		{
			SIGM* aux = (SIGM*) malloc(sizeof(SIGM));

			aux->batch_size = INTEGER(getListElement(VECTOR_ELT(layers, i), "batch_size"))[0];
			aux->n_units = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_units"))[0];

			aux->a = gsl_matrix_calloc(aux->batch_size, aux->n_units);

			pipeline[i].type = 9;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "DIRE") == 0)
		{
			DIRE* aux = (DIRE*) malloc(sizeof(DIRE));

			aux->batch_size = INTEGER(getListElement(VECTOR_ELT(layers, i), "batch_size"))[0];
			aux->n_units = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_units"))[0];

			aux->buff_x = gsl_matrix_calloc(aux->batch_size, aux->n_units);
			aux->buff_dy = gsl_matrix_calloc(aux->batch_size, aux->n_units);

			pipeline[i].type = 11;
			pipeline[i].layer = (void*) aux;
		}
		else if (strcmp(s, "TANH") == 0)
		{
			TANH* aux = (TANH*) malloc(sizeof(TANH));

			aux->batch_size = INTEGER(getListElement(VECTOR_ELT(layers, i), "batch_size"))[0];
			aux->n_units = INTEGER(getListElement(VECTOR_ELT(layers, i), "n_units"))[0];

			aux->a = gsl_matrix_calloc(aux->batch_size, aux->n_units);

			pipeline[i].type = 12;
			pipeline[i].layer = (void*) aux;
		}
		else // CELL, MSEL, others...
		{
			pipeline[i].type = 0;
			pipeline[i].layer = NULL; // Let the system explode, for now...
		}
	}

	return pipeline;
}

// Interface for Predicting and Reconstructing using a CNN
SEXP _C_CNN_predict (SEXP newdata, SEXP layers, SEXP num_layers)
{
 	int nrows = INTEGER(GET_DIM(newdata))[0];
 	int nchan = INTEGER(GET_DIM(newdata))[1];
 	int img_h = INTEGER(GET_DIM(newdata))[2];
 	int img_w = INTEGER(GET_DIM(newdata))[3];

 	int nlay = INTEGER_VALUE(num_layers);

	// Re-assemble the CNN (build pipeline)
	LAYER* pipeline = reassemble_CNN(layers, nlay);

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
	gsl_matrix* predictions = prediction_cnn (test_X, nrows, nchan, pipeline, nlay);
	int nouts = (int) predictions->size2;

	// Prepare Results
	SEXP retval = PROTECT(allocMatrix(REALSXP, nrows, nouts));
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < nouts; j++)
			REAL(retval)[i * nouts + j] = gsl_matrix_get(predictions, i, j);

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

// Interface for Training a MLP
SEXP _C_MLP_train (SEXP dataset, SEXP targets, SEXP layers, SEXP num_layers, SEXP batch_size,
	SEXP training_epochs, SEXP learning_rate, SEXP momentum, SEXP rand_seed)
{
 	int nrows = INTEGER(GET_DIM(dataset))[0];
 	int ncols = INTEGER(GET_DIM(dataset))[1];

 	int nouts = INTEGER(GET_DIM(targets))[1];

 	int basi = INTEGER_VALUE(batch_size);
 	int trep = INTEGER_VALUE(training_epochs);
 	int rase = INTEGER_VALUE(rand_seed);
 	double lera = NUMERIC_VALUE(learning_rate);
 	double mome = NUMERIC_VALUE(momentum);

	int nlays = INTEGER_VALUE(num_layers);

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

	// Build the Layers pipeline
	LAYER* pipeline = build_pipeline(layers, nlays);

	// Train a CNN
	double loss = train_mlp(train_X, train_Y, pipeline, nlays, trep, basi, lera, mome, rase);

	// Pass the Training set through the MLP
	gsl_matrix* predictions = prediction_mlp(train_X, pipeline, nlays);
	gsl_matrix* confusion = classification_matrix(predictions, train_Y);

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
			REAL(VECTOR_ELT(retval, 4))[i * nouts + j] = gsl_matrix_get(confusion, i, j);

	SET_VECTOR_ELT(retval, 5, allocVector(REALSXP, 1));
	REAL(VECTOR_ELT(retval, 5))[0] = loss;

	SET_VECTOR_ELT(retval, 6, allocMatrix(REALSXP, nrows, nouts));
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < nouts; j++)
			REAL(VECTOR_ELT(retval, 6))[i * nouts + j] = gsl_matrix_get(predictions, i, j);

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

	gsl_matrix_free(train_X);
	gsl_matrix_free(train_Y);
	gsl_matrix_free(predictions);
	gsl_matrix_free(confusion);

	return retval;
}

// Interface for Predicting and Reconstructing using a MLP
SEXP _C_MLP_predict (SEXP newdata, SEXP layers, SEXP num_layers)
{
 	int nrows = INTEGER(GET_DIM(newdata))[0];
 	int ncols = INTEGER(GET_DIM(newdata))[1];

 	int nlay = INTEGER_VALUE(num_layers);

	// Re-assemble the CNN (build pipeline)
	LAYER* pipeline = reassemble_CNN(layers, nlay);

	// Prepare Test Dataset
	gsl_matrix* test_X = gsl_matrix_alloc(nrows, ncols);
	for (int r = 0; r < nrows; r++)
		for (int c = 0; c < ncols; c++)
			gsl_matrix_set(test_X, r, c, RMATRIX(newdata, r, c));

	// Pass through CNN
	gsl_matrix* predictions = prediction_mlp (test_X, pipeline, nlay);
	int nouts = (int) predictions->size2;

	// Prepare Results
	SEXP retval = PROTECT(allocMatrix(REALSXP, nrows, nouts));
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < nouts; j++)
			REAL(retval)[i * nouts + j] = gsl_matrix_get(predictions, i, j);

	// Free the structures and the CNN
	free_pipeline (pipeline, nlay);

	gsl_matrix_free(test_X);
	gsl_matrix_free(predictions);

	UNPROTECT(1);

	return retval;
}

