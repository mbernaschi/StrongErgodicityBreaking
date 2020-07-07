#include <stdlib.h>
#include <stdio.h>

#define MY_CUDA_CHECK( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define MY_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }

void writelog(int, int, const char *, ...);
#define MAKEMATR_RC 1
#if !defined(TRUE)
enum {FALSE, TRUE};
#endif
#if !defined(MAKEMATR_RC) 
#define MAKEMATR_RC 12
#endif

void **mmcuda(void ***rp, int r, int c, int s, int init) {
  int i;
  char **pc;
  short int **psi;
  int **pi;
  double **pd;
  char **d_pc;
  short int **d_psi;
  int **d_pi;
  double **d_pd;


  switch(s) {
  case sizeof(char):
    pc=(char **)malloc(r*sizeof(char *));
    if(!pc) writelog(TRUE, MAKEMATR_RC, "error in makematr 1\n");
    MY_CUDA_CHECK( cudaMalloc( (void **) &d_pc, r*sizeof(char*) ) );
    for(i=0; i<r; i++) {
      MY_CUDA_CHECK( cudaMalloc( (void **) &pc[i], c*sizeof(char) ) );
      if(init) {
            MY_CUDA_CHECK( cudaMemset( pc[i], 0, c*sizeof(char) ) );
      }
    }
    MY_CUDA_CHECK( cudaMemcpy( d_pc, pc, r*sizeof(char *), cudaMemcpyHostToDevice ) );
    rp[0]=(void **)d_pc;
    return (void **)pc;
  case sizeof(short int):
    psi=(short int **)malloc(r*sizeof(short int*));
    if(!psi) writelog(TRUE, MAKEMATR_RC, "error in makematr 2\n");
    MY_CUDA_CHECK( cudaMalloc( (void **) &d_psi, r*sizeof(short int*) ) );
    for(i=0; i<r; i++) {
      MY_CUDA_CHECK( cudaMalloc( (void **) &psi[i], c*sizeof(short int) ) );
      if(init) {
            MY_CUDA_CHECK( cudaMemset( psi[i], 0, c*sizeof(short int) ) );
      }
    }
    MY_CUDA_CHECK( cudaMemcpy( d_psi, psi, r*sizeof(short int*), cudaMemcpyHostToDevice ) );
    rp[0]=(void **)d_psi;
    return (void **)psi;
  case sizeof(int):
    pi=(int **)malloc(r*sizeof(int*));
    if(!pi) writelog(TRUE, MAKEMATR_RC, "error in makematr 3\n");
    MY_CUDA_CHECK( cudaMalloc( (void **) &d_pi, r*sizeof(int*) ) );
    for(i=0; i<r; i++) {
      MY_CUDA_CHECK( cudaMalloc( (void **) &pi[i], c*sizeof(int) ) );
      if(init) {
            MY_CUDA_CHECK( cudaMemset( pi[i], 0, c*sizeof(int) ) );
      }
    }
    MY_CUDA_CHECK( cudaMemcpy( d_pi, pi, r*sizeof(int *), cudaMemcpyHostToDevice ) );
    rp[0]=(void **)d_pi;
    return (void **)pi;
  case sizeof(double):
    pd=(double **)malloc(r*sizeof(double*));
    if(!pd) writelog(TRUE, MAKEMATR_RC, "error in makematr 4 for %d rows\n",r);
    MY_CUDA_CHECK( cudaMalloc( (void **) &d_pd, r*sizeof(double*) ) );
    for(i=0; i<r; i++) {
      MY_CUDA_CHECK( cudaMalloc( (void **) &pd[i], c*sizeof(double) ) );
      if(init) {
            MY_CUDA_CHECK( cudaMemset( pd[i], 0, c*sizeof(double) ) );
      }
    }
    MY_CUDA_CHECK( cudaMemcpy( d_pd, pd, r*sizeof(double *), cudaMemcpyHostToDevice ) );
    rp[0]=(void **)d_pd;
    return (void **)pd;
  default:
    writelog(TRUE,MAKEMATR_RC,"Unexpected size: %d\n",s);
    break;
  }
  return NULL;
}
