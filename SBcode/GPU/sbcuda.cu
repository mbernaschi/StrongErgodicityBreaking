/* by FRT, CUDA porting by MB */
// time ./a.out 16 .8 30 15 128 1 1234567
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#include  <assert.h>
#include <curand_kernel.h>
#include  <iostream>
#include  <map>
#include  <vector>
#include "timing.h"
#include "common.h"
#ifdef USE_LDG
#define LDG(x) (__ldg(&(x)))
#else
#define LDG(x) (x)
#endif

using namespace std;
// COMPILA con -fopenmp
//#include "omp.h"
enum{MAX=65536};
//enum{M=1024,MASK=M-1};
enum{M=1,MASK=M-1};
//enum{M=8,MASK=M-1};
#define DEGREE 4
#define DEGREE2 2

#define NDIR 2
#define DEVICONST __device__ __constant__
#define DEVIBASE __device__
#define CUDATHREADS 1024
#define THREADS 1
#define SPT 4
#define DUMPFREQ 20480
#define MAXFILENM 1024
#define DEFAULTNGPU 1
#define MAXNGPU 4
#define NGPUENVV "NGPU_X_BETHE"
#define MAXNCHUNK 128
#define NCHUNKENVV "NCHUNK_X_BETHE"

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
enum {mult=16807, mod=2147483647};

void **mmcuda(void ***rp, int, int, int, int);

typedef unsigned long long int MYWORD;
struct Vicini {
    MYWORD J[DEGREE*M];
    int neig[DEGREE];
} ;

typedef struct {
        int t;
        int salvati;
} tmds;

typedef struct {
        int t;
        MYWORD *p2s;
} disds;

static int ngpu=1;

void *takeMeasures(void *);
int dumpSpin(int, MYWORD *);
void *dumpItwSpin(void *);
int dumpVicini(struct Vicini *, struct Vicini *);
int loadVicini(char *, struct Vicini *, struct Vicini *);
int loadSpin(char *, MYWORD *, int);
char prefixname[]="spingraph";
char *restartvicinifile=NULL;
char *restartspinfile=NULL;
int dump=TRUE;
int toffset=0;

DEVIBASE struct Vicini *d_vicini;
DEVICONST unsigned int d_prob2, d_prob4;
DEVICONST MYWORD d_allOnes;
DEVIBASE curandState  *randstates[MAXNGPU];
unsigned long long *h_randstates;
static int nrandstates=0;

__global__ void initRand(unsigned int seed, curandState *state, unsigned int n){

  const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

  /* Each thread gets same seed , a different sequence number  -  no offset */
  if(tid<n) {
    curand_init (seed , tid , 0 , &state [ tid ]) ;
  }

}


int sum[MAX];
void InitBinario(){
    assert(sizeof(long long int)==4*sizeof(short));
    for (int cou=0;cou<MAX;cou++){
        int scra=cou;
        int ris=0;
        for(int i=0; i<16;i++){
            ris+=scra&1;
            scra>>=1;
        }
        sum[cou]=ris;
        //cout<<cou<<" "<<ris<<endl;
    }
};
int Sum(unsigned long long op){
    union Hack {unsigned long long lungo;  unsigned short corto[4];} hack;
    hack.lungo=op;
    //cerr<<hack.corto[0]<<" "<<hack.corto[1]<<" "<<hack.corto[2]<<" "<<hack.corto[3]<<endl;
    return  sum[hack.corto[0]]+sum[hack.corto[1]]+sum[hack.corto[2]]+sum[hack.corto[3]];
}

struct RandomC{
    enum {mult=16807, mod=2147483647};
    unsigned long long seed;
    int IntRand(){
        seed = (mult * seed) % mod;
        return int(seed);
    }
    RandomC(unsigned int op=1234567){
        assert(op>0);
        seed=op;
    }
};
RandomC randPar[THREADS];


__global__ void d_oneMCstepBN(MYWORD *, MYWORD *, int, MYWORD **, int **, int,
                              curandState *, int, int, int, int);

#define MYWORD_LEN (int(8 * sizeof(MYWORD)))
#define FNORM   (2.3283064365e-10)
#define RANDOM  ((ira[ip++] = ira[ip1++] + ira[ip2++]) ^ ira[ip3++])
#define FRANDOM (FNORM * RANDOM)
#define pm1 ((FRANDOM > 0.5) ? 1 : -1)

Vicini * viciniB,* viciniN;
int size, ntw, *graph;
double beta;
unsigned int prob2, prob4;
MYWORD *dueAlla, allZeros, allOnes, *spin, **stw;
int tdump=DUMPFREQ;

/* variabili globali per il generatore random */
unsigned int myrand, ira[256];
unsigned char ip, ip1, ip2, ip3;

unsigned int randForInit(void) {
    unsigned long long int y;

    y = myrand * 16807LL;
    myrand = (unsigned int)((y & 0x7fffffff) + (y >> 31));
    if (myrand & 0x80000000) {
        myrand = (myrand & 0x7fffffff) + 1;
    }
    return myrand;
}

void initRandom(void) {
    int i;

    ip = 128;
    ip1 = ip - 24;
    ip2 = ip - 55;
    ip3 = ip - 61;

    for (i = ip3; i < ip; i++) {
        ira[i] = randForInit();
    }
    for(i = 0; i < THREADS; i++)
        randPar[i]=RandomC(RANDOM);
}

MYWORD randomWord(void) {
    int i;
    MYWORD res = 0;

    for (i = 0; i < MYWORD_LEN; i++) {
        res = res << 1;
        if (FRANDOM < 0.5) res++;
    }
    return res;
}

void error(const char *string) {
    fprintf(stderr, "ERROR: %s\n", string);
    exit(EXIT_FAILURE);
}

void printBinary(MYWORD word) {
    int i;

    for (i = MYWORD_LEN - 1; i >= 0; i--) {
        if (word & dueAlla[i])
            printf("1");
        else
            printf("0");
    }
    printf("\n");
}

void checkTypes(void) {
    if (sizeof(unsigned short int) != 2) {
        error("lo short int non ha 16 bits!");
    }
    printf("# le parole sono a %i bits\n", (int)MYWORD_LEN);
}

void initDueAlla(void) {
    int i;

    allZeros = (MYWORD)0;
    allOnes = ~allZeros;
    dueAlla = (MYWORD *)calloc(MYWORD_LEN, sizeof(MYWORD));
    for (i = 0; i < MYWORD_LEN; i++) {
        dueAlla[i] = (MYWORD)1 << i;
    }
}

int initGraph(void) {
    for (int volte=0;volte<DEGREE;volte++){
        int numUnchosen = size;
        int volteBack=(volte+DEGREE2)&(DEGREE-1);
        vector<int> unchosen (numUnchosen);
        for (int i = 0; i <numUnchosen; i++) unchosen[i] = i;
        for(int i=0;i<size;i++){
            int index = (int)(FRANDOM * numUnchosen);
            int picked = unchosen[index];
            unchosen[index] = unchosen[--numUnchosen];
            viciniB[i].neig[volte] = picked;
            viciniN[picked].neig[volteBack]=i;
            for(int x=0;x<M;x++){
                MYWORD coup = randomWord();
                viciniB[i].J[volte+x*DEGREE] = coup;
                int y=(x-volte/DEGREE2)&MASK;
                viciniN[picked].J[volteBack+x*DEGREE /* volteBack+y*DEGREE */]=coup;
            }
        }
        assert(numUnchosen==0);
    }
    if(dump) {
      if(dumpVicini(viciniB, viciniN)) {
        fprintf(stderr,"Error in dumpVicini\n");
        exit(1);
      }
    }
    for (int i = 0; i < 2*M*size; i++)spin[i] = randomWord();
    return 0;
}

void oneMCstepBN(MYWORD * __restrict__ newSpin, MYWORD * __restrict__ oldSpin,const int dir) {
    // Questa funzione vale solo se DEGREE == 4
    const int dirBack=dir-1;
    Vicini * __restrict__ vicini= (dir==0)?viciniB:viciniN;
#pragma omp parallel default(shared) num_threads(THREADS)
    {
#pragma omp for schedule(static)
        for(int k = 0; k < THREADS; k++){
            const int lower = (size * k) / THREADS;
            const int upper = (size * (k+1)) / THREADS;
            for (int i = lower; i < upper; i++) {
                const int base=i*M;
                const int y=dir&MASK;
                const int yBack=dirBack&MASK;
                for(int x=0;x<M;x++){
                    const int point=base+x;
                    const int vicino0=M*vicini[i].neig[0]+y;
                    const int vicino1=M*vicini[i].neig[1]+y;
                    const int vicino2=M*vicini[i].neig[2]+yBack;
                    const int vicino3=M*vicini[i].neig[3]+yBack;
                    MYWORD t1 = newSpin[point]^ oldSpin[vicino0] ^ vicini[i].J[0+x*DEGREE];
                    MYWORD t2 = newSpin[point]^ oldSpin[vicino1] ^ vicini[i].J[1+x*DEGREE];
                    MYWORD t3 = newSpin[point]^ oldSpin[vicino2] ^ vicini[i].J[2+x*DEGREE];
                    MYWORD t4 = newSpin[point]^ oldSpin[vicino3] ^ vicini[i].J[3+x*DEGREE];
                    unsigned int ran = randPar[k].IntRand();
                  MYWORD condition0 = -(ran>prob2);
                  MYWORD condition1 = -(ran>prob4);
                  MYWORD flipBit0 = ((t1 ^ t2) & (t3 ^ t4)) | (t1 & t2) | (t3 & t4);
                  MYWORD flipBit1 = t1 | t2 | t3 | t4;
                  MYWORD flipBit = (condition0 & flipBit0) |
                  (~condition0 & condition1 & flipBit1) |
                  (~condition1);
                  newSpin[point] ^= flipBit;
                }
            }
        }
    }

}

__device__ unsigned int GPUIntRand(unsigned long long *pts) {
        pts[0]=(mult * pts[0]) % mod;
        return int(pts[0]);
}

#define GETRAN(s,r) (s)=(mult * (s)) % mod;\
                       (r)=int((s));
    //                 asm volatile("mov.b64 {%0,%1}, %2;":"=r"(r),"=r"(scratch):"l"(s));



__global__ void
__launch_bounds__(1024, 1)
d_oneMCstepBN(MYWORD * __restrict__ newSpin,
                                  MYWORD * __restrict__ oldSpin,
                                  const int dir,
                                  MYWORD **J,
                                  int **neig,
                                  int n,
                                  curandState *state,
                                  int gpu_id, int ngpu, int chunk_id, int nchunk) {

    // Questa funzione vale solo se DEGREE == 4
    // const int dirBack=dir-1;
    unsigned int ran;
    const unsigned int tid = threadIdx.x+blockDim.x*blockIdx.x;
    const unsigned int tthreads = gridDim.x*blockDim.x;
    int i;
    int spinxgpu=n/ngpu;
    int spinxchunk=spinxgpu/nchunk;
    curandState localState=state[tid];
    for(int x=0;x<M;x++){




                const int base=x*n;
                for(i=tid+(gpu_id*spinxgpu)+(chunk_id*spinxchunk);i<((gpu_id)*spinxgpu)+((chunk_id+1)*spinxchunk);i+=tthreads) {
                    //const int y=0 /* dir&MASK */;
                    //const int yBack=0 /* dirBack&MASK */;
                    const int point=base+i;
                    const int vicino0=base+neig[DEGREE*dir+0][i];
                    const int vicino1=base+neig[DEGREE*dir+1][i];
                    const int vicino2=base+neig[DEGREE*dir+2][i];
                    const int vicino3=base+neig[DEGREE*dir+3][i];
		    ran = curand(&localState);
//                  MYWORD t0 = LDG(newSpin[point]);
                    MYWORD t1 = LDG(newSpin[point])^oldSpin[vicino0]^LDG(J[0+DEGREE*(x+M*dir)][i]);
                    MYWORD t2 = LDG(newSpin[point])^oldSpin[vicino1]^LDG(J[1+DEGREE*(x+M*dir)][i]);
                    MYWORD t3 = LDG(newSpin[point])^oldSpin[vicino2]^LDG(J[2+DEGREE*(x+M*dir)][i]);
                    MYWORD t4 = LDG(newSpin[point])^oldSpin[vicino3]^LDG(J[3+DEGREE*(x+M*dir)][i]);
#if 0
                    if(base==0) {
                      printf("dir=%d, point=%d, spin=%llu, vicino0=%d,vicino1=%d,vicino2=%d,vicino3=%d,J[0]=%llu,J[1]=%llu,J[2]=%llu,J[3]=%llu\n",dir,point,newSpin[point],vicino0, vicino1, vicino2, vicino3, J[0+DEGREE*(x+M*dir)][i],J[1+DEGREE*(x+M*dir)][i],J[2+DEGREE*(x+M*dir)][i],J[3+DEGREE*(x+M*dir)][i]);
                    }
#endif
                    MYWORD condition0 = -(ran>d_prob2);
                    MYWORD condition1 = -(ran>d_prob4);
                    MYWORD flipBit0 = ((t1 ^ t2) & (t3 ^ t4)) | (t1 & t2) | (t3 & t4);
                    MYWORD flipBit1 = t1 | t2 | t3 | t4;
                    MYWORD flipBit = (condition0 & flipBit0) |
                    (~condition0 & condition1 & flipBit1) |
                    (~condition1);
                    newSpin[point] ^= flipBit;
                }
    }
    state[tid]=localState;
}


void printMeanVar(double *x, int num) {
    int i;
    double mean=0.0, var=0.0;

    for (i = 0; i< num; i++) {
        mean += x[i];
        var += x[i] * x[i];
    }
    mean /= num;
    var = var / num - mean * mean;
    printf(" %g %g", mean, sqrt(var/num));
}

void *dumpItwSpin(void *vp) {
  disds *p=(disds *)vp;
  int  t=p->t;
  MYWORD *spin=p->p2s;
  char filename[MAXFILENM];
  FILE *fpd;
  int nitems=2*size*M;
  snprintf(filename,sizeof(filename),"itw_%s_%d.dump",prefixname,t);
  fpd=Fopen(filename,"w");
  if(fwrite(spin,sizeof(MYWORD), nitems, fpd)<nitems) {
    perror("dumpItwSpin: ");
  }
  fclose(fpd);
  return NULL;
}

int dumpSpin(int t, MYWORD *s) {
  static char *lastname=NULL;
  char filename[MAXFILENM];
  FILE *fpd;
  int nitems=2*size*M;
  snprintf(filename,sizeof(filename),"%s_%d.dump",prefixname,t);
  fpd=Fopen(filename,"w");
  if(fwrite(s,sizeof(MYWORD), nitems, fpd)<nitems) {
    perror("dumpSpin: ");
  }
  if(fwrite(h_randstates,sizeof(unsigned long long), nrandstates*ngpu, fpd)<(nrandstates*ngpu)) {
    perror("dumpSpin 2: ");
  }
  fclose(fpd);
  if(lastname) {
    if(unlink(lastname)<0) {
      fprintf(stderr,"Error removing %s\n",lastname);
    }
    free(lastname);
  }
  lastname=Strdup(filename);
  return 0;
}

int loadSpin(char *filename, MYWORD *s, int lrn) {
  FILE *fpd;
  int nitems=2*size*M;
  fpd=Fopen(filename,"r");
  if(fread(s,sizeof(MYWORD), nitems, fpd)<nitems) {
    perror("loadSpin: ");
    return -1;
  }
  if(lrn) {
  if(fread(h_randstates,sizeof(unsigned long long), nrandstates*ngpu, fpd)<(nrandstates*ngpu)) {
    perror("loadSpin 2: ");
  }
  for(int i=0; i<ngpu; i++) {
    MY_CUDA_CHECK( cudaMemcpy(randstates[i], &(h_randstates[i*nrandstates]),
                 sizeof(unsigned long long)*nrandstates, cudaMemcpyHostToDevice) );
  }
  }
  fclose(fpd);
  return 0;
}


void *takeMeasures(void *vp) {
    int t, salvati;
    int i, j, itw;
    double invSize = 1./(size*M);
    MYWORD tmp;
    unsigned long long ene=0;
    tmds *p=(tmds *)vp;
    t=p->t;
    salvati=p->salvati;
    printf("%i", t);
    /*    for (i = 0; i < size; i++) {
     for (j = 0; j < DEGREE; j++) {
     tmp = spin[i] ^ spin[size+vicini[i].neig[j]] ^ vicini[i].J[j];
     ene+=Sum(tmp);
     }
     }*/
    for (i = 0; i < size; i++) {
        int base=i;
        for(int x=0;x<M;x++){
            int yBack=(x-1)&MASK;
            for (j = 0; j < DEGREE2; j++) {
                MYWORD tmp =spin[base+x*size] ^
                  spin[((M+x)*size)+viciniB[i].neig[j]] ^ viciniB[i].J[j+x*DEGREE];
                ene+=Sum(tmp);
                int j2=j+DEGREE2;
                tmp =spin[base+x*size] ^
                  spin[((M+x)*size)+(viciniB[i].neig[j2])] ^ viciniB[i].J[j2+x*DEGREE];
                ene+=Sum(tmp);
            }
        }
    }

    printf(" %g  \t",invSize/MYWORD_LEN*ene- 0.5 * DEGREE);
    for (itw = 0; itw < salvati; itw++) {
        unsigned long long int remanent=0;
        for (i = 0; i < 2*size*M; i++) {
            tmp = spin[i] ^ stw[itw][i];
            remanent+=Sum(tmp);
        }
        printf(" %g",1.0-invSize/MYWORD_LEN*remanent );
    }
    printf("\n");
    fflush(stdout);
    if(dump && t>0 && ((t%tdump)==0)) {
      if(dumpSpin(t,spin)) {
        fprintf(stderr,"Error dumping spin!\n");
      }
    }
    return NULL;
}

int dumpVicini(struct Vicini *B, struct Vicini *N) {
  char filename[MAXFILENM];
  FILE *fpd;
  int nitems=size;
  snprintf(filename,sizeof(filename),"%s_vicini.dump",prefixname);
  fpd=Fopen(filename,"w");
  if(fwrite(B,sizeof(struct Vicini), nitems, fpd)<nitems) {
    perror("dumpVicini 1: ");
    return -1;
  }
  if(fwrite(N,sizeof(struct Vicini), nitems, fpd)<nitems) {
    perror("dumpVicini 2: ");
    return -1;
  }
  fclose(fpd);
  return 0;
}

int loadVicini(char *filename, struct Vicini *B, struct Vicini *N) {
  FILE *fpd;
  int nitems=size;
  fpd=Fopen(filename,"r");
  if(fread(B,sizeof(struct Vicini), nitems, fpd)<nitems) {
    perror("loadVicini 1: ");
    return -1;
  }
  if(fread(N,sizeof(struct Vicini), nitems, fpd)<nitems) {
    perror("loadVicini 2: ");
    return -1;
  }
  fclose(fpd);
  return 0;
}


int main(int argc, char *argv[]) {
    int logSize, logIter, numIter, numSamples;
    int is, c, i, j, k, d, measTime, itw;

    int nthreads=CUDATHREADS;
    int nblocks;
    int spt=SPT;

    MYWORD **d_J[MAXNGPU];
    int **d_neig[MAXNGPU];

    MYWORD **h_J[MAXNGPU];
    int **h_neig[MAXNGPU];

    MYWORD *jtemp;
    int *neigtemp;

    MYWORD **h_spin[MAXNGPU];
    MYWORD  **d_spin[MAXNGPU];
    cudaStream_t *stream;

    Vicini * vicini;

    double tempRatio;
    tmds *p2tmds;
    disds *p2disds;
    pthread_t ptid;
    pthread_t ptid2;
    static int tmt=0;
    static int dist=0;
    char *restartsitwfile=NULL;
    int baseitw;
    int rc;
    int nchunk=1;

    size_t freeMem, totMem;
    TIMER_DEF;
    //FILE *devran = fopen("/dev/random","r");
    FILE *devran = fopen("/dev/urandom","r"); // lower quality
    fread(&myrand, 4, 1, devran);
    fclose(devran);

    if(getenv(NGPUENVV)!=NULL) {
        ngpu=atoi(getenv(NGPUENVV));
        if(ngpu<1||ngpu>MAXNGPU) {
                fprintf(stderr,"Invalid number of gpu: %d, must be >0 and <%d\n", ngpu,MAXNGPU+1);
                exit(1);
        }
    }
    if(getenv(NCHUNKENVV)!=NULL) {
        nchunk=atoi(getenv(NCHUNKENVV));
        if(nchunk<1||nchunk>MAXNCHUNK) {
                fprintf(stderr,"Invalid number of chunk: %d, must be >0 and <%d\n", nchunk,MAXNCHUNK+1);
                exit(1);
        }
    }

    if (argc < 7) {
        fprintf(stderr, "usage: %s <logSize> <T/Tc> <logIter> <log(last_tw)> <measTime> <numSamples> [<seed>]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    printf("# SG +/-J on a RRG with d = %i\n", DEGREE);
    logSize = atoi(argv[1]);
    size = 1 << logSize;
    printf("# bipartite RRG of size  = %i M %i n", 2*size,M);
    tempRatio = atof(argv[2]);
    beta = atanh(1/sqrt(3.)) / tempRatio;
    prob2 = (unsigned int)(exp(-4.*beta) * 2 * RandomC::mod);
    prob4 = (unsigned int)(exp(-8.*beta) * 2 * RandomC::mod);
    printf("# T = %f Tc   beta = %g   prob2 = %u   prob4 = %u\n", tempRatio, beta, prob2, prob4);
    logIter = atoi(argv[3]);
    numIter = 1 << logIter;
    printf("# logIter = %i   numIter = %i\n", logIter, numIter);
    ntw = atoi(argv[4]) + 1;
    printf("# 1 <= tw <= %i in potenze di 2\n", 1<<(ntw-1));
    measTime = atoi(argv[5]);
    printf("# measTime = %i\n", measTime);
    numSamples = atoi(argv[6]);
    printf("# numSamples = %i\n", numSamples);
    if (argc > 7) myrand = atoi(argv[7]);
    printf("# seed = %u\n", myrand);
    if (argc > 8) {
       restartvicinifile = Strdup(argv[8]);
       restartspinfile = Strdup(argv[9]);
       baseitw=10;
    }
    int trueSize=2*size*M;
    viciniB = (struct Vicini *)calloc(size, sizeof(struct Vicini));
    viciniN = (struct Vicini *)calloc(size, sizeof(struct Vicini));
    spin = (MYWORD *)calloc(trueSize, sizeof(MYWORD));
    stw = (MYWORD **)calloc(ntw+2, sizeof(MYWORD *));
    for (i = 0; i < ntw+2; i++)
        stw[i] = (MYWORD *)calloc(trueSize, sizeof(MYWORD));
    checkTypes();
    initRandom();
    InitBinario();
    initDueAlla();
    p2tmds=(tmds *)calloc(1,sizeof(tmds));
    if(p2tmds==NULL) {
       fprintf(stderr,"Could not get memory for pointer to tmds\n");
       exit(1);
    }
    p2disds=(disds *)calloc(1,sizeof(disds));
    if(p2disds==NULL) {
       fprintf(stderr,"Could not get memory for pointer to disds\n");
       exit(1);
    }
    printf("# 1:t  2:ener  (3,4):q  (5,6):|q|  (7,8):q^2  9+log(tw)/2:C(t,tw)\n");
    MYWORD *bianchi=spin;
    MYWORD *neri= spin+ M*size;
    nblocks=((size/ngpu/nchunk)+CUDATHREADS-1)/CUDATHREADS/spt;
    nrandstates=nblocks*nthreads;
    cudaMemGetInfo(&freeMem, &totMem);
    printf("#Using %d gpus, %d chunks\n",ngpu, nchunk);
    stream = (cudaStream_t*) Malloc(2*ngpu * sizeof(cudaStream_t));
    for(int g=0; g<ngpu; g++) {
      MY_CUDA_CHECK(cudaSetDevice( g ));
      MY_CUDA_CHECK( cudaStreamCreate(&(stream[g])) );
      MY_CUDA_CHECK( cudaStreamCreate(&(stream[g+ngpu])) );
      printf("#%llu bytes of GPU global memory are available on gpu %d\n",freeMem, g);
      MY_CUDA_CHECK( cudaMemcpyToSymbol(d_prob2,&prob2,sizeof(unsigned int),0,cudaMemcpyHostToDevice) );
      MY_CUDA_CHECK( cudaMemcpyToSymbol(d_prob4,&prob4,sizeof(unsigned int),0,cudaMemcpyHostToDevice) );
      MY_CUDA_CHECK( cudaMemcpyToSymbol(d_allOnes,&allOnes,sizeof(MYWORD),0,cudaMemcpyHostToDevice) );
      h_spin[g]=(MYWORD **)mmcuda((void ***)&d_spin[g],2,size*M,sizeof(MYWORD),1);
      h_J[g]=(MYWORD **)mmcuda((void ***)&d_J[g],2*DEGREE*M,size,sizeof(MYWORD),1);
      h_neig[g]=(int **)mmcuda((void ***)&d_neig[g],2*DEGREE,size,sizeof(int),1);
      MY_CUDA_CHECK ( cudaMalloc (( void **) &randstates[g], nrandstates*sizeof ( curandState ) ));
      cudaMemGetInfo(&freeMem, &totMem);
      printf("#Occupied %llu bytes of GPU global memory on gpu %d\n",totMem-freeMem, g);
      initRand<<<nblocks,nthreads>>>(myrand+(g*nrandstates),randstates[g],size);
      MY_CUDA_CHECK( cudaDeviceSynchronize() );
      cudaFuncSetCacheConfig(d_oneMCstepBN, cudaFuncCachePreferL1 );
      for(int h=0; h<ngpu; h++) {
        if(h==g) continue;
        MY_CUDA_CHECK(cudaDeviceEnablePeerAccess( h, 0 ) );
      }
    }
// Verifico quanto spazio ho in memoria
    jtemp=(MYWORD *)Malloc(sizeof(MYWORD)*size);
    neigtemp=(int *)Malloc(sizeof(int)*size);
    h_randstates=(unsigned long long *)Malloc(sizeof(curandState)*nrandstates*ngpu);

    for (is = 0; is < numSamples; is++) {
        printf("# Sample %i  Building graph.", is);
        if(restartspinfile==NULL) {
          while(initGraph()) printf(".");
        } else {
          if(loadVicini(restartvicinifile, viciniB, viciniN)!=0) {
             fprintf(stderr,"Error loading Vicini from file %s\n",restartvicinifile);
             exit(1);
          }
          if(loadSpin(restartspinfile, spin, 1)!=0) {
            fprintf(stderr,"Error loading Spin from file %s\n",restartspinfile);
            exit(1);
          }
          sscanf(restartspinfile,"spingraph_%d.dump",&toffset);
          printf("\n#restart done from iteration %d...",toffset);
        }
        printf("OK\n");
        fflush(stdout);
        for(int g=0; g<ngpu; g++) {
          MY_CUDA_CHECK(cudaSetDevice( g ));
          MY_CUDA_CHECK( cudaMemcpy(h_spin[g][0], bianchi,
                              sizeof(MYWORD)*size*M, cudaMemcpyHostToDevice) );
          MY_CUDA_CHECK( cudaMemcpy(h_spin[g][1], neri,
                              sizeof(MYWORD)*size*M, cudaMemcpyHostToDevice) );
        }
        for(d=0; d<NDIR; d++) {
          vicini = (d==0)?viciniB:viciniN;
          for(i=0; i<DEGREE; i++) {
            for(k=0; k<M; k++) {
              for(j=0; j<size; j++) {
                jtemp[j]=vicini[j].J[(i+k*DEGREE)];
              }
              for(int l=0; l<ngpu; l++) {
                MY_CUDA_CHECK(cudaSetDevice( l ));
                MY_CUDA_CHECK( cudaMemcpy(h_J[l][i+k*DEGREE+M*d*DEGREE], jtemp,
                                          sizeof(MYWORD)*size, cudaMemcpyHostToDevice) );
              }
            }
            for(j=0; j<size; j++) {
              neigtemp[j]=vicini[j].neig[i];
            }
            for(int l=0; l<ngpu; l++) {
              MY_CUDA_CHECK(cudaSetDevice( l ));
              MY_CUDA_CHECK( cudaMemcpy(h_neig[l][i+d*DEGREE], neigtemp,
                                      sizeof(int)*size, cudaMemcpyHostToDevice) );
            }
          }
        }
        free(jtemp);
        free(neigtemp);

        itw = 1;
        if(toffset==0) {
           p2tmds->t=0;
           p2tmds->salvati=itw;
           if(pthread_create(&ptid,NULL,takeMeasures,p2tmds)<0) {
              fprintf(stderr,"Error creating thread for taking measures\n");
              exit(1);
           }
           tmt=1;
           for (i = 0; i < trueSize; i++ ) stw[0][i] = spin[i];
           p2disds->t=0;
           p2disds->p2s=stw[0];
           if(pthread_create(&ptid2,NULL,dumpItwSpin,p2disds)<0) {
              fprintf(stderr,"Error creating thread for dumping spins itw\n");
              exit(1);
           }
           dist=1;
        } else {
          itw=0;
          for(j=baseitw; j<argc; j++) {
            if(restartsitwfile) free(restartsitwfile);
            restartsitwfile=Strdup(argv[j]);
            if(loadSpin(restartsitwfile, stw[j-baseitw], 0)!=0) {
                fprintf(stderr,"Error loading Spin %d from file %s\n",j-baseitw,restartsitwfile);
                exit(1);
            }
            printf("#reload itw file %s in stw[%d] ...\n",restartsitwfile,j-baseitw);
            itw++;
          }
//        p2tmds->t=toffset;
//        p2tmds->salvati=itw;
//        takeMeasures(p2tmds);
          //itw=(itw<(j-baseitw)?(j-baseitw):itw);
        }
        TIMER_START;
        for (int t=1+toffset; t <= numIter; t++) {
          for(c=0; c<nchunk; c++) {
            for(int g=0; g<ngpu; g++) {
              MY_CUDA_CHECK(cudaSetDevice( g ));
              d_oneMCstepBN<<<nblocks,nthreads, 0, stream[g]>>>(h_spin[g][0],h_spin[g][1],0,
                                                                d_J[g],d_neig[g],size,randstates[g], g, ngpu, c, nchunk);
            }
            for(int g=0; g<ngpu; g++) {
              MY_CUDA_CHECK(cudaStreamSynchronize(stream[g]));
            }
            for(int g=0; g<ngpu; g++) {
              for(int l=0; l<ngpu; l++) {
                if(l==g) { continue; }
                MY_CUDA_CHECK( cudaMemcpyPeerAsync(h_spin[l][0]+g*(size*M/ngpu)+c*(size*M/ngpu/nchunk),l,
                                                   h_spin[g][0]+g*(size*M/ngpu)+c*(size*M/ngpu/nchunk),g,
                                                   size/ngpu/nchunk*M*sizeof(MYWORD),stream[ngpu+g]) );
              }
            }
          }
          for(int g=0; g<ngpu; g++) {
              MY_CUDA_CHECK(cudaStreamSynchronize(stream[ngpu+g]));
          }
          for(c=0; c<nchunk; c++) {
            for(int g=0; g<ngpu; g++) {
              MY_CUDA_CHECK(cudaSetDevice( g ));
              d_oneMCstepBN<<<nblocks,nthreads, 0, stream[g]>>>(h_spin[g][1],h_spin[g][0],1,
                                                                d_J[g],d_neig[g],size,randstates[g], g, ngpu, c, nchunk);
            }
            for(int g=0; g<ngpu; g++) {
              MY_CUDA_CHECK(cudaStreamSynchronize(stream[g]));
            }
            for(int g=0; g<ngpu; g++) {
              for(int l=0; l<ngpu; l++) {
                if(l==g) { continue; }
                MY_CUDA_CHECK( cudaMemcpyPeerAsync(h_spin[l][1]+g*(size*M/ngpu)+c*(size*M/ngpu/nchunk),l,
                                                   h_spin[g][1]+g*(size*M/ngpu)+c*(size*M/ngpu/nchunk),g,
                                                   size/ngpu/nchunk*M*sizeof(MYWORD),stream[ngpu+g]) );
              }
            }
          }
          for(int g=0; g<ngpu; g++) {
              MY_CUDA_CHECK(cudaStreamSynchronize(stream[ngpu+g]));
          }

            //            oneMCstepBN(bianchi, neri,0);
            //  takeMeasures(t,itw);

            //            oneMCstepBN(neri, bianchi,1);
            //  takeMeasures(t,itw);

            if ((t == (1<<(2*(itw)))) && (itw<ntw)) {
                if(dist && (rc=pthread_join(ptid2,NULL)<0)) {
                  fprintf(stderr,"Error in join thread %d:\n",rc);
                  exit(1);
                }
                MY_CUDA_CHECK(cudaSetDevice( 0 ));
                MY_CUDA_CHECK( cudaMemcpy(stw[itw], h_spin[0][0],
                              sizeof(MYWORD)*size*M, cudaMemcpyDeviceToHost) );
                MY_CUDA_CHECK( cudaMemcpy(&(stw[itw][size*M]), h_spin[0][1],
                              sizeof(MYWORD)*size*M, cudaMemcpyDeviceToHost) );

                p2disds->t=itw;
                p2disds->p2s=stw[itw];
                if(pthread_create(&ptid2,NULL,dumpItwSpin,p2disds)<0) {
                fprintf(stderr,"Error creating thread for dumping spins itw\n");
                 exit(1);
                }
                // for (i = 0; i < trueSize; i++ ) stw[itw][i] = spin[i];
                itw++;
            }
            if (t % measTime == 0) {
                if(tmt && (rc=pthread_join(ptid,NULL)<0)) {
                  fprintf(stderr,"Error in join thread %d:\n",rc);
                  exit(1);
                }
                MY_CUDA_CHECK(cudaSetDevice( 0 ));
                MY_CUDA_CHECK( cudaMemcpy(bianchi, h_spin[0][0],
                              sizeof(MYWORD)*size*M, cudaMemcpyDeviceToHost) );
                MY_CUDA_CHECK( cudaMemcpy(neri, h_spin[0][1],
                              sizeof(MYWORD)*size*M, cudaMemcpyDeviceToHost) );
                for(int g=0; g<ngpu; g++) {
                  MY_CUDA_CHECK(cudaSetDevice( g ));
                  MY_CUDA_CHECK( cudaMemcpy(&(h_randstates[g*nrandstates]), randstates[g], sizeof(curandState)*nrandstates, cudaMemcpyDeviceToHost) );
                }
                p2tmds->t=t;
                p2tmds->salvati=itw;
                if(pthread_create(&ptid,NULL,takeMeasures,p2tmds)<0) {
                  fprintf(stderr,"Error creating thread for taking measures\n");
                  exit(1);
                } else {
                  tmt=1;
                }
                // takeMeasures(p2tmds);
                // takeMeasures(t,itw);
            }
        }
        TIMER_STOP;
        if(rc=pthread_join(ptid,NULL)<0) {
                  fprintf(stderr,"Error in join thread %d:\n",rc);
                  exit(1);
        }
        printf("#Total time for main loop: %7.5f\n",TIMER_ELAPSED);
        //takeMeasures(numIter);
        //printf("\n");
    }
    return EXIT_SUCCESS;
}
