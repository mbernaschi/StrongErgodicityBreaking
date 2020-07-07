/* by FRT */
/* 09/05/09: tolgo loop di lunghezza 3 */
/* 11/05/09: metto 4 repliche */
/* 06/08/09: correzione a sumBit0 */
/* 24/08/09: correzione nel passo di swap */
/* 15/10/12: tolgo la tabella per gli offset */
// 26/07/16: parto da SG_Bethe_field_PT.c e tolgo il PT
// 28/07/16: tolgo il campo
// 05/11/16: tolgo s2 e metto diversi tw

/* by AM */
// 24/04/18: uso randForInit per l'update (MINSTD invece di PR), ma vedi sotto
// 26/04/18: modificato per openmp (bipartito)
// 28/04/18: modificato per usare di nuovo PR invece di MINSTD (molti generatori PR)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <omp.h>
#include <sys/syscall.h>
#include <assert.h>
#include <sched.h>

typedef unsigned long long int MYWORD;

#define DEGREE 4
//#define QLINK
#define MYWORD_LEN (8 * sizeof(MYWORD))
#define FNORM (5.42101086242752e-20)
//#define FNORM   (2.3283064365e-10)
//#define RANDOM  ((ira[ip++] = ira[ip1++] + ira[ip2++]) ^ ira[ip3++])
#define RANDOM(__wra) (( (__wra).ira[__wra.ip++] = (__wra).ira[(__wra).ip1++]+(__wra).ira[(__wra).ip2++] )^(__wra).ira[(__wra).ip3++])
#define FRANDOM(__wra) (FNORM * RANDOM(__wra))
#define pm1(__wra) ((FRANDOM(__wra) > 0.5) ? 1 : -1)

#ifndef NCPU
#define NCPU (1)  
#endif


struct multispin {
    MYWORD spin, J[DEGREE];
    struct multispin *neig[DEGREE];
} *s;

struct rngWheel {
    unsigned long long ira[256];
    unsigned char ip, ip1, ip2, ip3; 
};

int size, ntw, *graph;
double beta;
unsigned long long int prob2, prob4;
MYWORD *dueAlla, allZeros, allOnes, **stw;

/* variabili globali per il generatore random */
unsigned int myrand[1];
struct rngWheel* pr;
//unsigned char ip, ip1, ip2, ip3;

int* cpulist;
cpu_set_t mycpuset;
int ncpu;

#define PROB_MAX (0xffffffffffffffffULL)
#define randForInit(_k,_tmpR) {						\
	_tmpR = myrand[_k] * 16807LL;					\
	myrand[_k] = (unsigned int)((_tmpR & 0x7fffffff) + (_tmpR >> 31)); \
	if (myrand[_k] & 0x80000000)    myrand[_k] = (myrand[_k] & 0x7fffffff) + 1; \
    }

void initRandom(struct rngWheel* rng) {
    int i;
    unsigned long long int tmpR;

    rng->ip = 255;    
    rng->ip1 = rng->ip - 24;    
    rng->ip2 = rng->ip - 55;    
    rng->ip3 = rng->ip - 61;
  
    for (i = rng->ip3; i < rng->ip; i++) {
	randForInit(0,tmpR);
	rng->ira[i] = myrand[0]<<1;
    }

    for (i=0;i<1000;i++) tmpR=RANDOM((*rng));
}

MYWORD randomWord(struct rngWheel* rng) {
    int i;
    MYWORD res = 0;
  
    for (i = 0; i < MYWORD_LEN; i++) {
	res = res << 1;
	if (FRANDOM((*rng)) < 0.5) res++;
    }
    return res;
}

void error(char *string) {
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

int initGraph(struct rngWheel* rng) {
    int i, j, *unchosenA, *unchosenB, numUnchosenA, numUnchosenB;
    int index, *deg, t, tMax;
    MYWORD coup;
  
#define halfsize (size>>1)

    deg = (int *)calloc(size, sizeof(int));
    unchosenA = (int *)calloc(DEGREE * halfsize, sizeof(int));
    unchosenB = (int *)calloc(DEGREE * halfsize, sizeof(int));
    for (i = 0; i < DEGREE * halfsize; i++) {
	unchosenA[i] = i % halfsize;
	unchosenB[i] = halfsize+i % halfsize;
    }
    numUnchosenA = numUnchosenB = DEGREE * halfsize;
    //    k = 0;
    while (numUnchosenA || numUnchosenB) {
	index = (int)(FRANDOM((*rng)) * numUnchosenA);
	i = unchosenA[index];
	unchosenA[index] = unchosenA[--numUnchosenA];
	t = 0;
	tMax = 10 * numUnchosenB;
	do {
	    index = (int)(FRANDOM((*rng)) * numUnchosenB);
	    t++;
	} while (i == unchosenB[index] && t < tMax); /*I modify accordingly but irrelevant now*/
	if (t == tMax) {
	    /*Should never happen, actually*/
	    free(unchosenA);
	    free(unchosenB);
	    return 1;
	}
	j = unchosenB[index];
	unchosenB[index] = unchosenB[--numUnchosenB];
    
	coup = randomWord(pr);
	//coup = allZeros;
	s[i].neig[deg[i]] = s + j;
	s[j].neig[deg[j]] = s + i;
	s[i].J[deg[i]] = coup;
	s[j].J[deg[j]] = coup;
	deg[i]++;
	deg[j]++;
#ifdef QLINK
	graph[k++] = i;
	graph[k++] = j;
#endif
    }
    if (numUnchosenA != 0 || numUnchosenA != numUnchosenB )  error("in graph generation! (wrong numUnchosen{A,B})");
    for (i = 0; i < size; i++) {
	if (deg[i] != DEGREE) error("in graph generation! (wrong deg[])");
	s[i].spin = randomWord(pr);
    }
    free(unchosenA);
    free(unchosenB);
    free(deg);
    return 0;
}

void oneMCstep(struct multispin *sp, int starti, int lasti, struct rngWheel* rng) {
    // Questa funzione vale solo se DEGREE == 4 
    int i;
    unsigned long long int ran;
    MYWORD t1, t2, t3, t4, flipBit;

    for (i = starti; i < lasti; i++) {
	//      assert(i<size);
	ran = RANDOM((*rng));
	t1 = (sp[i].spin) ^ (sp[i].J[0]) ^ ((sp[i].neig[0])->spin);
	t2 = (sp[i].spin) ^ (sp[i].J[1]) ^ ((sp[i].neig[1])->spin);
	t3 = (sp[i].spin) ^ (sp[i].J[2]) ^ ((sp[i].neig[2])->spin);
	t4 = (sp[i].spin) ^ (sp[i].J[3]) ^ ((sp[i].neig[3])->spin);
	if (ran > prob2) {
	    flipBit = ((t1 ^ t2) & (t3 ^ t4)) | (t1 & t2) | (t3 & t4);
	} else if (ran > prob4) {
	    flipBit = t1 | t2 | t3 | t4;
	} else {
	    flipBit = allOnes;
	}
	sp[i].spin ^= flipBit;
    }
  
}

void oneMCstep_wrapper(struct multispin *sp) {


    int tt, starti, lasti, spinblock, tid, icpu;
    omp_set_dynamic(0);
    omp_set_num_threads(NCPU);

    spinblock=halfsize/NCPU;  
#pragma omp parallel private(tt, starti, lasti, icpu)
    {

	tt=omp_get_thread_num();

#ifdef _SCHED_
	tid=(pid_t)syscall(SYS_gettid);
	sched_setaffinity(tid,sizeof(cpu_set_t),&mycpuset);
#ifdef _SCHED_TRACK_
	icpu=sched_getcpu():
	fprintf(stderr,"THREAD #%d (TID %d) running in cpu #%d\n",tt,tid,icpu);
#endif

#endif

	starti = spinblock * tt ; 
	lasti=( (tt==(NCPU-1)) ? halfsize : spinblock * (tt+1) );

	oneMCstep(sp,starti,lasti,pr+tt);
#pragma omp barrier
	starti+=halfsize; lasti+=halfsize;
	oneMCstep(sp,starti,lasti,pr+tt);

    }

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

void takeMeasures(int t) {
    int i, j, k, itw;
    int ener[MYWORD_LEN]={0};
    int q[MYWORD_LEN];
#ifdef QLINK
    int qLink[MYWORD_LEN];
#endif
    double obs[MYWORD_LEN], invSize = 1./size;
    MYWORD tmp;

    printf("%i", t);
    for (i = 0; i < size; i++) {
	for (j = 0; j < DEGREE; j++) {
	    tmp = (s[i].spin) ^ (s[i].J[j]) ^ ((s[i].neig[j])->spin);
	    for (k = 0; k < MYWORD_LEN; k++) {
		ener[k] += (int)(tmp & (MYWORD)1);
		tmp = tmp >> 1;
	    }
	}
    }
    for (k = 0; k < MYWORD_LEN; k++)
	obs[k] = invSize * ener[k] - 0.5 * DEGREE + 0.5 * DEGREE / sqrt(DEGREE - 1);
    printMeanVar(obs, MYWORD_LEN);
  
    for (itw = 0; itw < ntw; itw++) {
	for (k = 0; k < MYWORD_LEN; k++) {
	    q[k] = 0;
#ifdef QLINK
	    qLink[k] = 0;
#endif
	}
	for (i = 0; i < size; i++) {
	    tmp = s[i].spin ^ stw[itw][i];
	    for (k = 0; k < MYWORD_LEN; k++) {
		q[k] += (int)(tmp & (MYWORD)1);
		tmp = tmp >> 1;
	    }
	}
	for (k = 0; k < MYWORD_LEN; k++)
	    obs[k] = 1.0 - 2.0 * invSize * q[k];
	printMeanVar(obs, MYWORD_LEN);
    
#ifdef QLINK
	for (i = 0; i < size*DEGREE; i += 2) {
	    tmp = (s[graph[i]].spin) ^ (s[graph[i+1]].spin) ^
		stw[itw][graph[i]] ^ stw[itw][graph[i+1]];
	    for (k = 0; k < MYWORD_LEN; k++) {
		qLink[k] += (int)(tmp & (MYWORD)1);
		tmp = tmp >> 1;
	    }
	}
	for (k = 0; k < MYWORD_LEN; k++)
	    obs[k] = 1.0 - 2.0 * invSize / DEGREE * qLink[k];
	printMeanVar(obs, MYWORD_LEN);
#endif
    }
    printf("\n");
    fflush(stdout);
}

int read_cpu_list(int** cpulist, char* filename) {

    int nreg=1;
    int n;
    FILE* fin;

    *cpulist=(int*)malloc(nreg*sizeof(int));
    if (NULL==(fin=fopen(filename,"r"))) return 0;

    for(n=0;EOF!=fscanf(fin,"%d",(*cpulist)+n);n++) {
	if(n==(nreg-1)) {
	    nreg*=2; *cpulist=(int*)realloc(*cpulist,nreg*sizeof(int));      
	}
	if(n==1023) {
	    fprintf(stderr,"Something wrong with cpu input.\n");
	    exit(15);
	}
    }

    *cpulist=(int*)realloc(*cpulist,n*sizeof(int));

    fclose(fin);

    return n;

}

int main(int argc, char *argv[]) {

    int logSize, logIter, numIter, numSamples;
    int is, i, t, measTime, itw, tt, tid;
    double tempRatio;
    //FILE *devran = fopen("/dev/random","r");
    FILE *devran ;
    char* cpufile;
  
    if (argc != 7 && argc != 8) {
	fprintf(stderr, "usage: %s <logSize> <T/Tc> <logIter> <log(last_tw)> <measTime> <numSamples> [<file lista cpu>]\n", argv[0]);
	exit(EXIT_FAILURE);
    }

#ifdef _SCHED_

    if (argc == 8) cpufile = argv[7];
    else cpufile="cpu.list";
  
    /*Assicuriamoci che il thread principale giri su una CPU nel set scelto*/
    ncpu=read_cpu_list(&cpulist,cpufile);
    if(ncpu<NCPU) error("Non abbastanza cpu nel file cpu.list\n");

    __CPU_ZERO_S(sizeof(cpu_set_t),&mycpuset); 
    for (tt=0;tt<NCPU;tt++) { 
	__CPU_SET_S(cpulist[tt],sizeof(cpu_set_t),&mycpuset); 
    }

    tid=(pid_t)syscall(SYS_gettid);
    sched_setaffinity(tid,sizeof(cpu_set_t),&mycpuset);

#endif

    /**/

    printf("# SG +/-J on a RRG with d = %i\n", DEGREE);
    logSize = atoi(argv[1]);
    size = 1 << logSize;
    printf("# logSize = %i   size = %i\n", logSize, size);
    tempRatio = atof(argv[2]);
    beta = atanh(1/sqrt(3.)) / tempRatio;
    prob2 = (unsigned long long int)(exp(-4.*beta) * PROB_MAX);
    prob4 = (unsigned long long int)(exp(-8.*beta) * PROB_MAX);
    printf("# T = %f Tc   beta = %g   prob2 = %llu   prob4 = %llu\n", tempRatio, beta, (unsigned long long)prob2, (unsigned long long)prob4);
    logIter = atoi(argv[3]);
    numIter = 1 << logIter;
    printf("# logIter = %i   numIter = %i\n", logIter, numIter);
    ntw = atoi(argv[4]) + 1;
    printf("# 1 <= tw <= %i in potenze di 2\n", 1<<(ntw-1));
    measTime = atoi(argv[5]);
    printf("# measTime = %i\n", measTime);
    numSamples = atoi(argv[6]);
    printf("# numSamples = %i\n", numSamples);

#ifdef QLINK
    graph = (int *)calloc(size*DEGREE, sizeof(int));
#endif
    pr = (struct rngWheel*)calloc(NCPU,sizeof(struct rngWheel));
    devran = fopen("/dev/urandom","r"); // lower quality
    fread(myrand, 4, 1, devran);
    fclose(devran);
    s = (struct multispin *)calloc(size, sizeof(struct multispin));
    stw = (MYWORD **)calloc(ntw+1, sizeof(MYWORD *));
    for (i = 0; i < ntw; i++)
	stw[i] = (MYWORD *)calloc(size, sizeof(MYWORD));
    checkTypes();
    for (i=0;i<NCPU;i++) initRandom(pr+i);
    initDueAlla();
#ifdef QLINK
    printf("# 1: t  (2,3): e(t)-e_c  (4,5)+4*log(tw): C(t,tw)  (6,7)+4*log(tw): Clink(t,tw)\n");
#else
    printf("# 1: t  (2,3): e(t)-e_c  (4,5)+2*log(tw): C(t,tw)\n");
#endif

    for (is = 0; is < numSamples; is++) {
	printf("# Sample %i  Building graph.", is);
	while(initGraph(pr)) printf(".");
	printf("OK\n");
	fflush(stdout);
	itw = 0;
	for (t = 1; t <= (1<<(ntw-1)); t++) {
	    oneMCstep_wrapper(s);
	    if (t == (1<<itw)) {
		for (i = 0; i < size; i++ )
		    stw[itw][i] = s[i].spin;
		itw++;
	    }
	}
	for (; t <= numIter; t++) {
	    oneMCstep_wrapper(s);
	    if (t % measTime == 0)
		takeMeasures(t);
	}
	takeMeasures(numIter);
	printf("\n");
    }
    return EXIT_SUCCESS;
}
