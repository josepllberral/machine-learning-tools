/*---------------------------------------------------------------------------*/
/* RESTRICTED BOLTZMANN MACHINES in C                                        */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// Original Code:
// https://github.com/josepllberral/machine-learning-tools

// Inspired by the implementations from:
// * David Buchaca   : https://github.com/davidbp/connectionist
// * Andrew Landgraf : https://www.r-bloggers.com/restricted-boltzmann-machines-in-r/
// * Graham Taylor   : http://www.uoguelph.ca/~gwtaylor/
// * Yusuke Sugomori : https://github.com/yusugomori/DeepLearning/

// Compile using "gcc -c rbm.c matrix_ops.c -lgsl -lgslcblas -lm -o rbm"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include "matrix_ops.h"

#ifndef RBM_H
#define RBM_H 1

typedef struct {
	int N;
	int n_visible;
	int n_hidden;
	gsl_matrix* W;
	gsl_vector* hbias;
	gsl_vector* vbias;
	gsl_matrix* vel_W;
	gsl_vector* vel_v;
	gsl_vector* vel_h;
	int batch_size;
} RBM;

void create_RBM (RBM* rbm, int N, int n_visible, int n_hidden, int batch_size, gsl_matrix* W, gsl_vector* hbias, gsl_vector* vbias);
void free_RBM (RBM* rbm);
void visible_state_to_hidden_probabilities_rbm (RBM* rbm, gsl_matrix* v_sample, gsl_matrix** h_mean, gsl_matrix** h_sample);
void hidden_state_to_visible_probabilities_rbm (RBM* rbm, gsl_matrix* h_sample, gsl_matrix** v_mean, gsl_matrix** v_sample);
double cdk_RBM (RBM* rbm, gsl_matrix* input, double lr, double momentum, int k);
void train_rbm (RBM* rbm, gsl_matrix* batchdata, int nrow, int ncol, int batch_size, int n_hidden, int training_epochs, double learning_rate, double momentum, int rand_seed, gsl_matrix* init_W, gsl_vector* init_hbias, gsl_vector* init_vbias);
void reconstruct_RBM (RBM* rbm, gsl_matrix* pv_sample, gsl_matrix** activations, gsl_matrix** reconstruct);
void forward_RBM(RBM*, gsl_matrix*, gsl_matrix**);
void backward_RBM(RBM*, gsl_matrix*, gsl_matrix**);

#endif
