
## Requires:  GSL + OpenBLAS
## - apt-get install libgsl-dev libopenblas-base
## export OPENBLAS_NUM_THREADS=4
## export OMP_NUM_THREADS=4

CC = gcc
LIBS = -lgsl -lgslcblas -lm

all: rbm crbm

crbm:
	${CC} crbm.c matrix_ops.c ${LIBS} -o crbm

rbm:
	${CC} rbm.c matrix_ops.c ${LIBS} -o rbm

clean:
	rm -f *.o crbm rbm
