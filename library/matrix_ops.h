#include <math.h>

/*---------------------------------------------------------------------------*/
/* MATRIX OPERATIONS                                                         */
/*---------------------------------------------------------------------------*/

// Function for Matrix Multiplication
//   param A, B   : Matrices to Multiply (pointers)
//   param arow   : 1st dimension of A
//   param common : common dimension of A and B
//   param bcol   : 2nd dimension of B
double** matrix_product (double** A, double** B, int arow, int common, int bcol)
{
	double** C = (double**) malloc(sizeof(double*) * arow);
	for (int a = 0; a < arow; a++)
	{
		C[a] = (double*) malloc(sizeof(double) * bcol);
		for (int c = 0; c < bcol; c++)
		{
			C[a][c] = 0;
			for (int b = 0; b < common; b++) C[a][c] += A[a][b] * B[b][c];
		}
	}
	return C;
}

// Function for Matrix Sum with Matrix
//   param A, B   : Matrices to Sum (pointers)
//   param nrow   : 1st dimension of A and B
//   param ncol   : 2nd dimension of A and B
double** matrix_summat (double** A, double** B, int nrow, int ncol)
{
	double** C = (double**) malloc(sizeof(double*) * nrow);
	for (int a = 0; a < nrow; a++)
	{
		C[a] = (double*) malloc(sizeof(double) * ncol);
		for (int b = 0; b < ncol; b++) C[a][b] = A[a][b] + B[a][b];
	}
	return C;
}

// Function for Matrix Subtract from Matrix
//   param A, B   : Matrices to Subtract A - B (pointers)
//   param nrow   : 1st dimension of A and B
//   param ncol   : 2nd dimension of A and B
double** matrix_subtract (double** A, double** B, int nrow, int ncol)
{
	double** C = (double**) malloc(sizeof(double*) * nrow);
	for (int a = 0; a < nrow; a++)
	{
		C[a] = (double*) malloc(sizeof(double) * ncol);
		for (int b = 0; b < ncol; b++) C[a][b] = A[a][b] - B[a][b];
	}
	return C;
}

// Function for Matrix Sum with Vector
//   param A      : Matrix to Sum (pointer)
//   param B      : Vector to Sum (pointer)
//   param nrow   : 1st dimension of A and length of B
//   param ncol   : 2nd dimension of A
double** matrix_sumvec (double** A, double* B, int nrow, int ncol)
{
	double** C = (double**) malloc(sizeof(double*) * nrow);
	for (int a = 0; a < nrow; a++)
	{
		C[a] = (double*) malloc(sizeof(double) * ncol);
		for (int b = 0; b < ncol; b++) C[a][b] = A[a][b] + B[a];
	}
	return C;
}

// Function for Matrix Subtract from Vector
//   param A      : Target Matrix (pointer)
//   param B      : Vector to Subtract from (pointer)
//   param nrow   : 1st dimension of A and length of B
//   param ncol   : 2nd dimension of A
double** matrix_subtractvec (double** A, double* B, int nrow, int ncol)
{
	double** C = (double**) malloc(sizeof(double*) * nrow);
	for (int a = 0; a < nrow; a++)
	{
		C[a] = (double*) malloc(sizeof(double) * ncol);
		for (int b = 0; b < ncol; b++) C[a][b] = A[a][b] - B[a];
	}
	return C;
}

// Function for Matrix Multiplication with Scalar
//   param A      : Matrix to Multiply (pointer)
//   param S      : Scalar value
//   param nrow   : 1st dimension of A
//   param ncol   : 2nd dimension of A
double** matrix_scale (double** A, double B, int nrow, int ncol)
{
	double** C = (double**) malloc(sizeof(double*) * nrow);
	for (int a = 0; a < nrow; a++)
	{
		C[a] = (double*) malloc(sizeof(double) * ncol);
		for (int b = 0; b < ncol; b++) C[a][b] = A[a][b] * B;
	}
	return C;
}

// Function for Matrix Sum with Scalar
//   param A      : Matrix to Sum (pointer)
//   param S      : Scalar value
//   param nrow   : 1st dimension of A
//   param ncol   : 2nd dimension of A
double** matrix_addition (double** A, double B, int nrow, int ncol)
{
	double** C = (double**) malloc(sizeof(double*) * nrow);
	for (int a = 0; a < nrow; a++)
	{
		C[a] = (double*) malloc(sizeof(double) * ncol);
		for (int b = 0; b < ncol; b++) C[a][b] = A[a][b] + B;
	}
	return C;
}

// Function for Sigma over Matrix
//   param A      : Target Matrix (pointer)
//   param nrow   : 1st dimension of A
//   param ncol   : 2nd dimension of A
double** matrix_sigma (double** A, int nrow, int ncol)
{
	double** S = (double**) malloc(sizeof(double*) * nrow);
	for (int a = 0; a < nrow; a++)
	{
		S[a] = (double*) malloc(sizeof(double) * ncol);
		for (int b = 0; b < ncol; b++) S[a][b] = 1.0 / (1.0 + exp(A[a][b]));
	}
	return S;
}

// Function for Bernoulli Sampling over Matrix
//   param A      : Target Matrix (pointer)
//   param nrow   : 1st dimension of A
//   param ncol   : 2nd dimension of A
double** matrix_bernoulli (double** A, int nrow, int ncol)
{
	double** B = (double**) malloc(sizeof(double*) * nrow);
	for (int a = 0; a < nrow; a++)
	{
		B[a] = (double*) malloc(sizeof(double) * ncol);
		for (int b = 0; b < ncol; b++)
		{
			B[a][b] = 0;
			double r = rand() / (RAND_MAX + 1.0);
			if (A[a][b] >= 0 && A[a][b] <= 1 && r < A[a][b]) B[a][b] = 1;
		}
	}
	return B;
}

// Function for Square Difference over Matrices
//   param A, B   : Target Matrices for (A - B)^2 (pointer)
//   param nrow   : 1st dimension of A and B
//   param ncol   : 2nd dimension of A and B
double** matrix_sqdiff (double** A, double** B, int nrow, int ncol)
{
	double** S = (double**) malloc(sizeof(double*) * nrow);
	for (int a = 0; a < nrow; a++)
	{
		S[a] = (double*) malloc(sizeof(double) * ncol);
		for (int b = 0; b < ncol; b++) S[a][b] = pow(A[a][b] - B[a][b], 2);
	}
	return S;
}


// Function for Sum Rows of a Matrix
//   param A      : Target Matrix (pointer)
//   param nrow   : 1st dimension of A
//   param ncol   : 2nd dimension of A
double* matrix_sumrows (double** A, int nrow, int ncol)
{
	double* S = (double*) malloc(sizeof(double) * nrow);
	for (int a = 0; a < nrow; a++)
	{
		S[a] = 0;
		for (int b = 0; b < ncol; b++) S[a] += A[a][b];
	}
	return S;
}

// Function for Sum Cols of a Matrix
//   param A      : Target Matrix (pointer)
//   param nrow   : 1st dimension of A
//   param ncol   : 2nd dimension of A
double* matrix_sumcols (double** A, int nrow, int ncol)
{
	double* S = (double*) malloc(sizeof(double) * ncol);
	for (int b = 0; b < ncol; b++) S[b] = 0;
	for (int a = 0; a < nrow; a++)
		for (int b = 0; b < ncol; b++)
			S[b] += A[a][b];
	return S;
}

// Function for Mean Cols of a Matrix
//   param A      : Target Matrix (pointer)
//   param nrow   : 1st dimension of A
//   param ncol   : 2nd dimension of A
double* matrix_meancols (double** A, int nrow, int ncol)
{
	double* S = (double*) malloc(sizeof(double) * ncol);
	for (int b = 0; b < ncol; b++) S[b] = 0;
	for (int a = 0; a < nrow; a++)
		for (int b = 0; b < ncol; b++)
			S[b] += A[a][b];
	for (int b = 0; b < ncol; b++) S[b] = S[b] / nrow;
	return S;
}

// Function for Creating a Normal Matrix
//   param nrow   : 1st dimension of new Matrix
//   param ncol   : 2nd dimension of new Matrix
//   param mean   : Mean for Distribution
//   param stdev  : Standard Deviation for Distribution
double** matrix_normal(int nrow, int ncol, double mean, double stdev) 
{
	double** N = (double**) malloc(sizeof(double*) * nrow);
	for (int i = 0; i < nrow; i++)
	{
		N[i] = (double*) malloc(sizeof(double) * ncol);
		for (int j = 0; j < ncol; j++)
		{
			double rnd1 = (rand() + 1.0)/(RAND_MAX + 1.0);
			double rnd2 = (rand() + 1.0)/(RAND_MAX + 1.0);
			N[i][j] = mean + sqrt(-2 * log(rnd1)) * cos( 2 * M_PI * rnd2) / stdev;
		}
	}
	return N;
}

// Function for Creating a Zeros Matrix
//   param nrow   : 1st dimension of new Matrix
//   param ncol   : 2nd dimension of new Matrix
double** matrix_zeros (int nrow, int ncol)
{
	double** Z = (double**) malloc(sizeof(double*) * nrow);
	for (int i = 0; i < nrow; i++)
	{
		Z[i] = (double*) malloc(sizeof(double) * ncol);
		for (int j = 0; j < ncol; j++) Z[i][j] = 0;
	}
	return Z;
}

// Function to Transpose a Matrix
//   param A      : Target Matrix (pointer)
//   param nrow   : 1st dimension of A
//   param ncol   : 2nd dimension of A
double** matrix_transpose (double** A, int nrow, int ncol)
{
	double** T = (double**) malloc(sizeof(double*) * ncol);
	for (int i = 0; i < ncol; i++)
	{
		T[i] = (double*) malloc(sizeof(double) * nrow);
		for (int j = 0; j < nrow; j++) T[i][j] = A[j][i];
	}
	return T;
}

// Function to Copy a Matrix
//   param A      : Target Matrix (pointer)
//   param nrow   : 1st dimension of A
//   param ncol   : 2nd dimension of A
double** matrix_copy (double** A, int nrow, int ncol)
{
	double** C = (double**) malloc(sizeof(double*) * nrow);
	for (int i = 0; i < nrow; i++)
	{
		C[i] = (double*) malloc(sizeof(double) * ncol);
		for (int j = 0; j < ncol; j++) C[i][j] = A[i][j];
	}
	return C;
}

// Function to Free a Matrix
//   param A      : Target Matrix (pointer)
//   param nrow   : 1st dimension of A
void matrix_free (double** A, int nrow)
{
	for (int i = 0; i < nrow; i++) free(A[i]);
	free(A);

	return;
}

// Function to Free a Matrix and Replace with Another
//   param A      : Target Matrix (pointer)
//   param B      : New Matrix (pointer)
//   param nrow   : 1st dimension of A
void matrix_replace (double*** A, double** B, int nrow)
{
	matrix_free(*A, nrow);
	*A = B;
	
	return;
}

/*---------------------------------------------------------------------------*/
/* VECTOR OPERATIONS                                                         */
/*---------------------------------------------------------------------------*/

// Function for Vector Sum with Vector
//   param A      : Vector to Sum (pointer)
//   param B      : Vector to Sum (pointer)
//   param npos   : Length of A and B
double* vector_sumvec (double* A, double* B, int npos)
{
	double* S = (double*) malloc(sizeof(double) * npos);
	for (int a = 0; a < npos; a++) S[a] = A[a] + B[a];
	return S;
}

// Function for Creating a Zeros Vector
//   param npos   : Length of new Vector
double* vector_zeros (int npos)
{
	double* V = (double*) malloc(sizeof(double) * npos);
	for (int i = 0; i < npos; i++) V[i] = 0;
	return V;
}

// Function for Vector Multiplication with Scalar
//   param V      : Vector to Multiply (pointer)
//   param S      : Scalar value
//   param npos   : length of V
double* vector_scale (double* V, double S, int npos)
{
	double* R = (double*) malloc(sizeof(double) * npos);
	for (int a = 0; a < npos; a++) R[a] = V[a] * S;
	return R;
}

// Function to Free a Vector and Replace with Another
//   param V      : Target Vector (pointer)
//   param N      : New Vector (pointer)
void vector_replace (double** V, double* N)
{
	free(*V);
	*V = N;
	
	return;
}

// Function to find the Mean from a Vector
//   param V      : Target Vector (pointer)
//   param npos   : length of V
double vector_mean (double* V, int npos)
{
	double sum = 0;
	for (int i = 0; i < npos; i++) sum += V[i];
	return sum / npos;
}

