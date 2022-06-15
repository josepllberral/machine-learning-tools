/*---------------------------------------------------------------------------*/
/* CONVOLUTIONAL NETWORKS in C for R                                         */
/*---------------------------------------------------------------------------*/

// @author Josep Ll. Berral (Barcelona Supercomputing Center)

// @date 24th April 2017

// References:
// * Approach based on Lars Maaloee's:
//   https://github.com/davidbp/day2-Conv
// * Also from LeNet (deeplearning.net)
//   http://deeplearning.net/tutorial/lenet.html

// Compile using "gcc flat.c line.c mops.c msel.c relu.c sigm.c test.c cnn.c conv.c grad_check.c mlp.c pool.c relv.c soft.c dire.c tanh.c rbml.c xent.c -lgsl -lgslcblas -lm -o cnn"

// Information for "Type" attributes:
// Layer Type:
// 1: convolutional
// 2: pooling
// 3: rectifier linear (4D image version)
// 4: flattening
// 5: linear
// 6: softmax
// 7: cross-entropy
// 8: rectifier linear (2D matrix version)
// 9: sigmoid
// 10: mean-squared-error
// 11: direct buffer
// 12: hyperbolic tangent
// 13: gaussian bernoully RBM

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#ifndef CNN_H
#define CNN_H 1

typedef struct {
	int batch_size;
	int filter_size;
	int n_filters;
	int n_channels;
	gsl_matrix*** W;
	gsl_vector* b;
	gsl_matrix*** grad_W;
	gsl_vector* grad_b;
	int pad_y;
	int pad_x;
	int win_h;
	int win_w;
	gsl_matrix*** img;
} CONV;

typedef struct {
	int batch_size;
	int n_channels;
	int win_size;
	int stride;
	int padding;
	gsl_matrix*** img;
} POOL;

typedef struct {
	int batch_size;
	int n_channels;
	int img_h;
	int img_w;
} FLAT;

typedef struct {
	int batch_size;
	int n_channels;
	gsl_matrix*** img;
} RELU;

typedef struct {
	int batch_size;
	gsl_matrix* img;
} RELV;

typedef struct {
	int batch_size;
	int n_hidden;
	int n_visible;
	gsl_matrix* W;
	gsl_matrix* grad_W;
	gsl_vector* b;
	gsl_vector* grad_b;
	gsl_matrix* x;
} LINE;

typedef struct {
	int batch_size;
	int n_hidden;
	int n_visible;
	int n_gibbs;
	gsl_matrix* W;
	gsl_matrix* grad_W;
	gsl_vector* hbias;
	gsl_vector* grad_hbias;
	gsl_vector* vbias;
	gsl_vector* grad_vbias;
	gsl_matrix* x;
	gsl_matrix* ph_means;
	double loss;
} RBML;

typedef struct {
	int batch_size;
	int n_units;
} SOFT;

typedef struct {
	int batch_size;
	int n_units;
	gsl_matrix* a;
} SIGM;

typedef struct {
	int batch_size;
	int n_units;
	gsl_matrix* a;
} TANH;

typedef struct {
	int batch_size;
	int n_units;
	gsl_matrix* buff_x;
	gsl_matrix* buff_dy;
} DIRE;

typedef struct {
	double loss;
} XENT;

typedef struct {
	double loss;
} MSEL;

typedef struct {
	int type;
	void* layer;
} LAYER;

typedef union {
	gsl_matrix*** image;
	gsl_matrix* matrix;
} data;

// Auxiliar Functions
void replace_image (gsl_matrix****, gsl_matrix****, int, int);
void print_image00 (gsl_matrix***, int, int);
void classification_matrix_print (gsl_matrix* predicted, gsl_matrix* observed);

// Convolutional Auxiliar Functions
gsl_matrix* conv2D (gsl_matrix*, gsl_matrix*, int);
gsl_matrix* img_padding (gsl_matrix*, int, int);

// Convolutional Layer
gsl_matrix*** conv_op (gsl_matrix***, int, int, gsl_matrix***, int, int, int, int);
gsl_matrix*** forward_conv (CONV*, gsl_matrix***);
gsl_matrix*** backward_conv (CONV*, gsl_matrix***);
void get_updates_conv (CONV*, double);
void create_CONV (CONV*, int, int, int, double, int, int);
void free_CONV (CONV*);
void copy_CONV (CONV*, CONV*);
int compare_CONV (CONV*, CONV*);

// Pooling Layer
gsl_matrix*** forward_pool (POOL*, gsl_matrix***);
gsl_matrix*** backward_pool (POOL*, gsl_matrix***);
void get_updates_pool (POOL*, double);
void create_POOL (POOL*, int, double, int, int, int);
void free_POOL (POOL*);
void copy_POOL (POOL*, POOL*);
int compare_POOL (POOL*, POOL*);

// Flattening Layer
gsl_matrix* forward_flat (FLAT*, gsl_matrix***);
gsl_matrix*** backward_flat (FLAT*, gsl_matrix*);
void get_updates_flat (FLAT*, double);
void create_FLAT (FLAT*, int, int);
void free_FLAT (FLAT*);
void copy_FLAT (FLAT*, FLAT*);
int compare_FLAT (FLAT*, FLAT*);

// Rectified Linear Layer
gsl_matrix*** forward_relu (RELU*, gsl_matrix***);
gsl_matrix*** backward_relu (RELU*, gsl_matrix***);
void get_updates_relu (RELU*, double);
void create_RELU (RELU*, int, int);
void free_RELU (RELU*);
void copy_RELU (RELU*, RELU*);
int compare_RELU (RELU*, RELU*);

// Linear Layer
gsl_matrix* forward_line (LINE*, gsl_matrix*);
gsl_matrix* backward_line (LINE*, gsl_matrix*);
void get_updates_line (LINE*, double);
void create_LINE (LINE*, int, int, double, int);
void free_LINE (LINE*);
void copy_LINE (LINE*, LINE*);
int compare_LINE (LINE*, LINE*);

// GB-RBM Layer
gsl_matrix* forward_rbml (RBML*, gsl_matrix*);
gsl_matrix* backward_rbml (RBML*, gsl_matrix*);
gsl_matrix* evaluate_rbml (RBML*, gsl_matrix*, gsl_matrix*, double, double*, double*);
void get_updates_rbml (RBML*, double);
void create_RBML (RBML*, int, int, double, int, int);
void free_RBML (RBML*);
void copy_RBML (RBML*, RBML*);
int compare_RBML (RBML*, RBML*);

// SoftMax Layer
gsl_matrix* forward_soft (SOFT*, gsl_matrix*);
gsl_matrix* backward_soft (SOFT*, gsl_matrix*);
void get_updates_soft (SOFT*, double);
void create_SOFT (SOFT*, int, int);
void free_SOFT (SOFT*);
void copy_SOFT (SOFT*, SOFT*);
int compare_SOFT (SOFT*, SOFT*);

// Sigmoid Layer
gsl_matrix* forward_sigm (SIGM*, gsl_matrix*);
gsl_matrix* backward_sigm (SIGM*, gsl_matrix*);
void get_updates_sigm (SIGM*, double);
void create_SIGM (SIGM*, int, int);
void free_SIGM (SIGM*);
void copy_SIGM (SIGM*, SIGM*);
int compare_SIGM (SIGM*, SIGM*);

// Direct Layer
gsl_matrix* forward_dire (DIRE*, gsl_matrix*);
gsl_matrix* backward_dire (DIRE*, gsl_matrix*);
void get_updates_dire (DIRE*, double);
void create_DIRE (DIRE*, int, int);
void free_DIRE (DIRE*);
void copy_DIRE (DIRE*, DIRE*);
int compare_DIRE (DIRE*, DIRE*);

// HyperTan Layer
gsl_matrix* forward_tanh (TANH*, gsl_matrix*);
gsl_matrix* backward_tanh (TANH*, gsl_matrix*);
void get_updates_tanh (TANH*, double);
void create_TANH (TANH*, int, int);
void free_TANH (TANH*);
void copy_TANH (TANH*, TANH*);
int compare_TANH (TANH*, TANH*);

// Cross-Entropy Layer
gsl_matrix* forward_xent (XENT*, gsl_matrix*, gsl_matrix*);
gsl_matrix* backward_xent (XENT*, gsl_matrix*, gsl_matrix*);
gsl_matrix* evaluate_xent (XENT*, gsl_matrix*, gsl_matrix*, double, double*, double*);
void get_updates_xent (XENT*, double);
void create_XENT (XENT*);
void free_XENT (XENT*);
void copy_XENT (XENT*, XENT*);
int compare_XENT (XENT*, XENT*);

// Mean-Squared Error Layer
gsl_matrix* forward_msel (MSEL*, gsl_matrix*, gsl_matrix*);
gsl_matrix* backward_msel (MSEL*, gsl_matrix*, gsl_matrix*);
void get_updates_msel (MSEL*, double);
void create_MSEL (MSEL*);
void free_MSEL (MSEL*);
void copy_MSEL (MSEL*, MSEL*);
int compare_MSEL (MSEL*, MSEL*);

// Rectified Linear Layer - Matrix Versions
gsl_matrix* forward_relv (RELV*, gsl_matrix*);
gsl_matrix* backward_relv (RELV*, gsl_matrix*);
void get_updates_relv (RELV*, double);
void create_RELV (RELV*, int);
void free_RELV (RELV*);
void copy_RELV (RELV*, RELV*);
int compare_RELV (RELV*, RELV*);

// General Functions
void forward (LAYER*, data*, int*);
void backward (LAYER*, data*, int*);
void get_updates (LAYER*, double);
gsl_matrix* evaluate_loss (LAYER*, gsl_matrix*, gsl_matrix*, double, double*, double*);
double train_cnn (gsl_matrix***, gsl_matrix*, int, int, LAYER*, int, LAYER*, int, int, double, double, int);
double train_mlp (gsl_matrix*, gsl_matrix*, LAYER*, int, LAYER*, int, int, double, double, int);
gsl_matrix* prediction_mlp (gsl_matrix* , LAYER*, int, int);
gsl_matrix* prediction_cnn (gsl_matrix***, int, int, LAYER*, int, int);
double classification_accuracy (gsl_matrix*, gsl_matrix*);
gsl_matrix* classification_matrix (gsl_matrix*, gsl_matrix*);
void pass_through_cnn (gsl_matrix***, int, int, LAYER*, int, int, gsl_matrix**, gsl_matrix****);
void pass_through_mlp (gsl_matrix*, LAYER*, int, int, gsl_matrix**, gsl_matrix**);

// Gradiend Check Functions
int gradclose (gsl_matrix***, gsl_matrix***, int, int, double, double);
int gradclose_vec (gsl_vector*, gsl_vector*, int, double, double);

double fun (void*, gsl_matrix***, gsl_matrix***, int);
gsl_matrix*** approx_fprime (void*, gsl_matrix***, gsl_matrix***, double, int, int, int);
gsl_matrix*** fun_grad (void*, gsl_matrix***, int, int);

double fun_b_conv (CONV*, gsl_vector*, gsl_matrix***);
gsl_vector* approx_fprime_b_conv (CONV*, gsl_matrix***, double);
gsl_vector* fun_grad_b_conv (CONV*, gsl_matrix***);

int gradclose_line (gsl_matrix*, gsl_matrix*, double, double);
double fun_line (LINE*, gsl_matrix*, gsl_matrix*);
double fun_b_line (LINE*, gsl_vector*, gsl_matrix*);
gsl_matrix* fun_grad_line (LINE*, gsl_matrix*, int);
gsl_vector* fun_grad_b_line (LINE*, gsl_matrix*);
gsl_matrix* approx_fprime_line (LINE*, gsl_matrix*, gsl_matrix*, double);
gsl_vector* approx_fprime_b_line (LINE*, gsl_matrix*, double);

int check_grad_conv (CONV*, gsl_matrix***, int, double, double, double);
int check_grad_pool (POOL*, gsl_matrix***, int, double, double, double);
int check_grad_flat (FLAT*, gsl_matrix***, int, double, double, double);
int check_grad_relu (RELU*, gsl_matrix***, int, double, double, double);
int check_grad_line (LINE*, gsl_matrix*, int, double, double, double);
int check_grad_soft (SOFT*, gsl_matrix*, int, double, double, double);
int check_grad_xent (XENT*, gsl_matrix*, gsl_matrix*, int, double, double, double);

// Matrix Operations
gsl_matrix* matrix_normal (int, int, double, double, double);
void matrix_sigmoid (gsl_matrix*, gsl_matrix*);
void matrix_exponent (gsl_matrix*, gsl_matrix*);
void matrix_bernoulli (gsl_matrix*, gsl_matrix*);
void matrix_log (gsl_matrix*, gsl_matrix*);
double matrix_sum (gsl_matrix*);
int* sequence (int, int);
int* shuffle (int);

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

// Drivers for Layers
int main_conv();
int main_pool();
int main_flat();
int main_relu();
int main_line();
int main_soft();
int main_xent();
int main_cnn();
int main_mlp();

#endif
