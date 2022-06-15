
## Requires:  GSL + OpenBLAS
## - apt-get install libgsl-dev libopenblas-base
## export OPENBLAS_NUM_THREADS=4
## export OMP_NUM_THREADS=4

CC = gcc
LIBS = -lgsl -lgslcblas -lm

LAYERS = flat.c line.c msel.c relu.c sigm.c conv.c pool.c relv.c soft.c dire.c tanh.c rbml.c xent.c
NETWORKS = pipeline_cnn.c pipeline_mlp.c
LIBRARIES = operations.c

all: test cnn

test:
	${CC} ${LAYERS} ${LIBRARIES} grad_check.c test.c ${LIBS} -o test
	
cnn:
	${CC} ${LAYERS} ${NETWORKS} ${LIBRARIES} main.c ${LIBS} -o cnn

clean:
	rm -f *.o cnn test
