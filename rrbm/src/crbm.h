/*---------------------------------------------------------------------------*/
/* CONDITIONAL RESTRICTED BOLTZMANN MACHINES in C for R                      */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// Inspired by the implementations from:
// * David Buchaca   : https://github.com/davidbp/connectionist
// * Andrew Landgraf : https://www.r-bloggers.com/restricted-boltzmann-machines-in-r/
// * Graham Taylor   : http://www.uoguelph.ca/~gwtaylor/
// * Yusuke Sugomori : https://github.com/yusugomori/DeepLearning/

// References:
// * Approach based on Graham Taylor's CRBM:
//   http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/gwtaylor_nips.pdf

// Mocap data:
// * R converted version: (you should find it in in the same GIT)
// * Original file: http://www.uoguelph.ca/~gwtaylor/publications/nips2006mhmublv/motion.mat
// * Data originally from Eugene Hsu, MIT. http://people.csail.mit.edu/ehsu/work/sig05stf/

// Compile using "gcc -c crbm.c matrix_ops.c -lgsl -lgslcblas -lm -o crbm"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#include "matrix_ops.h"

#ifndef CRBM_H
#define CRBM_H 1

typedef struct {
	int N;
	int n_visible;
	int n_hidden;
	gsl_matrix* A;
	gsl_matrix* B;
	gsl_matrix* W;
	gsl_vector* hbias;
	gsl_vector* vbias;
	gsl_matrix* vel_A;
	gsl_matrix* vel_B;
	gsl_matrix* vel_W;
	gsl_vector* vel_v;
	gsl_vector* vel_h;
	int delay;
	int batch_size;
} CRBM;

void create_CRBM (CRBM* crbm, int N, int n_visible, int n_hidden, int delay, int batch_size, gsl_matrix* A, gsl_matrix* B, gsl_matrix* W, gsl_vector* hbias, gsl_vector* vbias);
void free_CRBM (CRBM* crbm);
void visible_state_to_hidden_probabilities_crbm (CRBM* crbm, gsl_matrix* v_sample, gsl_matrix* v_history, gsl_matrix** h_mean, gsl_matrix** h_sample);
void hidden_state_to_visible_probabilities_crbm (CRBM* crbm, gsl_matrix* h_sample, gsl_matrix* v_history, gsl_matrix** v_mean, gsl_matrix** v_sample);
double cdk_CRBM (CRBM* crbm, gsl_matrix* input, gsl_matrix* input_history, double lr, double momentum, int k);
void train_crbm (CRBM* crbm, gsl_matrix* batchdata, int* seqlen, int nseq, int nrow, int ncol, int batch_size, int n_hidden, int training_epochs, double learning_rate, double momentum, int delay, int rand_seed, gsl_matrix* init_A, gsl_matrix* init_B, gsl_matrix* init_W, gsl_vector* init_hbias, gsl_vector* init_vbias);
void reconstruct_CRBM (CRBM* crbm, gsl_matrix* v_sample, gsl_matrix** activations, gsl_matrix** reconstruct);
gsl_vector* sample_fn (CRBM* crbm, int n_gibbs, gsl_vector** vis_sample, gsl_matrix** v_history);
gsl_matrix* generate_samples (CRBM* crbm, gsl_matrix* sequence, int n_samples, int n_gibbs);

#endif
