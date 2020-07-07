CC=g++ -m64
LD=nvcc
CFLAGS= -c -g -O3 -I . -DLINUX
LDFLAGS= -lm
CUDADIR= /usr/local/cuda
CUDAINCLUDEDIR= $(CUDADIR)/include

all: sbcuda

sbcuda: sbcuda.o common.o mmcuda.o
	${LD} -o sbcuda sbcuda.o common.o mmcuda.o ${LDFLAGS} ${CUDALDFLAGS} -lpthread

sbcuda.o: sbcuda.cu
	nvcc -m64 -c -O3 sbcuda.cu -arch=sm_70 -Xptxas=-v

mmcuda.o: mmcuda.cu
	nvcc -m64 -c -O3 mmcuda.cu -arch=sm_70

.c.o:; $(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o sbcuda
