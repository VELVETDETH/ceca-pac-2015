#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <getopt.h>

#include "proj.h"
#include "proto.h"
#include "utility.h"

#define DEFAULT_NUM_THREADS 8

float g[NPROJ*NRAY+10];

float f[IMGSIZE*IMGSIZE+10];
float v[IMGSIZE*IMGSIZE+10];

float lambda;

void MSbeam();

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

struct Stopwatch {
	clockid_t clk_id_;
	struct timespec start_;
	struct timespec count_;
	enum { TICKING, STOPPED } state_;
	int num_calls;
};

struct Stopwatch timer_main_real;
struct Stopwatch timer_main_proc;

void reset(struct Stopwatch *timer)
{
	timer->count_.tv_sec = 0;
	timer->count_.tv_nsec = 0;
	timer->state_ = STOPPED;
	timer->num_calls = 0;
}

void init(struct Stopwatch *timer, clockid_t clk_id)
{
	timer->clk_id_ = clk_id;
	reset(timer);
}

void tick(struct Stopwatch *timer)
{
	if (timer->state_ != TICKING) {
		timer->state_ = TICKING;
		++timer->num_calls;
		clock_gettime(timer->clk_id_, &timer->start_);
	}
}

float seconds(struct Stopwatch *timer)
{
	if (timer->state_ == STOPPED) {
		return timer->count_.tv_sec + 1e-9 * timer->count_.tv_nsec;
	} else {
		struct timespec current;
		clock_gettime(timer->clk_id_, &current);
		long delta_sec = current.tv_sec - timer->start_.tv_sec;
		long delta_nsec = current.tv_nsec - timer->start_.tv_nsec;
		return (timer->count_.tv_sec + delta_sec)
			+ 1e-9 * (timer->count_.tv_nsec + delta_nsec);
	}
}

int main(int argc,char **argv) {
	init(&timer_main_real, CLOCK_REALTIME);
	init(&timer_main_proc, CLOCK_PROCESS_CPUTIME_ID);
	tick(&timer_main_real);
	tick(&timer_main_proc);

    int nthreads;
    char c;
    char *data_file_name = NULL;
    while ((c = getopt(argc, argv, "t:f:")) != -1) {
        switch (c) {
            case 't':
                nthreads = atoi(optarg);
                break;
            case 'f':
                data_file_name = optarg;
                break;
            default:
                fprintf(stderr, "Unrecognized option: %c\n", c);
                exit(1);
        }
    }
    // error checking
    if (nthreads == 0) {
        printf("nthreads hasn't been set, use default value %d ...\n", DEFAULT_NUM_THREADS);
        nthreads = DEFAULT_NUM_THREADS;
    }
    if (!data_file_name) {
        fprintf(stderr, "Can't use NULL data file name\n");
        exit(1);
    }

    printf("NUMBER OF THREADS=%d\n",nthreads);
    omp_set_num_threads(nthreads);

    printf("Running read_phantom ...\n");
    read_phantom(f, data_file_name);
    
    normalize(f);
    write_file(f, "std.dat");
    
    
	A(g, f);
    
    FILE *fout = fopen("g.dat","w");
    int i,j;
    for (i = 0; i<NPROJ; ++i) {
        for (j = 0; j<NRAY; ++j)
            fprintf(fout,"%f ",g[i*NRAY+j]);
        fprintf(fout,"\n");
    }
    fclose(fout);

    float start = seconds(&timer_main_real);

    MSbeam();

    float end = seconds(&timer_main_real);

    printf("TIME USED: %.2f\n",(float)(end-start));
    
    normalize(f);
    
    write_file(f, "img.dat");
    
    write_file(v, "edge.dat");
    
    return 0;
}


void minIMAGE(int np,int nr,int *line,float *weight,int numb,float snorm) {
    int i;

    float d[IMGSIZE*2+10];
    
    float Af = 0.;
    for (i = 0; i<numb; ++i) {
        int ind = line[i];
        Af += f[ind]*weight[i];
    }
    Af -= g[np*NRAY+nr];
    /*if (nr==383) printf("%f\n",Af);*/
    
    /* calculate div (v^2 \grad f) and Af-g*/
    for (i = 0; i<numb; ++i) { 
        int ind = line[i];
        int x = ind/IMGSIZE, y = ind%IMGSIZE;

        float tmp = 0.;
        float lap = 0.;
        
        if (x+1<IMGSIZE) tmp += sqr(v[ind])*(f[ind+IMGSIZE]-f[ind]);
        else             tmp += sqr(v[ind])*(   0          -f[ind]);
        
        if (y+1<IMGSIZE) tmp += sqr(v[ind])*(f[ind+1]-f[ind]);
        else             tmp += sqr(v[ind])*(   0    -f[ind]);
        
        if (x-1>=0)      tmp -= sqr(v[ind-IMGSIZE])*(f[ind]-f[ind-IMGSIZE]);
        else             tmp -=                     (f[ind]-0        );
        
        if (y-1>=0)      tmp -= sqr(v[ind-1])*(f[ind]-f[ind-1]);
        else             tmp -=               (f[ind]-0       );
        
        if (x+1<IMGSIZE) lap += f[ind+IMGSIZE];
        if (y+1<IMGSIZE) lap += f[ind+1];
        if (x-1>=0)      lap += f[ind-IMGSIZE];
        if (y-1>=0)      lap += f[ind-1];
        lap -= 4*f[ind];

        d[i] = -Af*weight[i]+ALPHA*(tmp+sqr(EPSILON)*lap);
    }
        
    for (i = 0; i<numb; ++i) {
        int ind = line[i];
        f[ind] += lambda*d[i];
        if (f[ind]<0) f[ind] = 0;
        if (f[ind]>255) f[ind] = 255;
    }
}

void minEDGE(int np,int nr,int *line,float *weight,int numb) {
    int i;

    float d[IMGSIZE*2+10];
    
    for (i = 0; i<numb; ++i) {
        int ind = line[i];
        int x = ind/IMGSIZE, y = ind%IMGSIZE;
        
        float a = 0.;
        float b = 0.;
        float c = 0.;

        if (x-1>=0)      a += sqr(f[ind]-f[ind-IMGSIZE]);
        else             a += sqr(f[ind]-0        );
        
        if (y-1>=0)      a += sqr(f[ind]-f[ind-1]);
        else             a += sqr(f[ind]-0       );
        
        a *= v[ind];
        
        b = v[ind]-1;
                
        if (x+1<IMGSIZE) c += v[ind+IMGSIZE];
        if (y+1<IMGSIZE) c += v[ind+1];
        if (x-1>=0)      c += v[ind-IMGSIZE];
        if (y-1>=0)      c += v[ind-1];
        c -= 4*v[ind];
        
        d[i] = -ALPHA*a-BETA/(4*EPSILON)*b+BETA*EPSILON*c;
    }
    
    for (i = 0; i<numb; ++i) {
        int ind = line[i];
        v[ind] += lambda*d[i];
        if (v[ind]<0) v[ind] = 0;
        if (v[ind]>1) v[ind] = 1;
    }
}

float AT_g[NPROJ*NRAY];

float AT,G,H;

void compute_AT(int flag) {
    A(AT_g, f);
    
    int i;
    float part_1 = 0., part_2 = 0., part_3 = 0.;
    
    for (i = 0; i<NPROJ*NRAY; ++i)
        part_1 += sqr(AT_g[i]-g[i]);
        
    for (i = 0; i<IMGSIZE*IMGSIZE; ++i) {
        float tmp = 0;
        float x = i/IMGSIZE, y = i%IMGSIZE;
        
        if (x-1 >= 0) tmp += sqr(f[i] - f[i-IMGSIZE]);
        else          tmp += sqr(f[i] - 0);
        if (y-1 >= 0) tmp += sqr(f[i] - f[i-1]);
        else          tmp += sqr(f[i] - 0);
        
        part_2 += tmp*sqr(v[i]);
    }
    part_2 *= ALPHA;
    
    for (i = 0; i<IMGSIZE*IMGSIZE; ++i) {
        float tmp = 0.;
        float x = i/IMGSIZE, y = i%IMGSIZE;
        
        if (x-1 >= 0) tmp += sqr(v[i] - v[i-IMGSIZE]);
        else          tmp += sqr(v[i] - 1);
        if (y-1 >= 0) tmp += sqr(v[i] - v[i-1]);
        else          tmp += sqr(v[i] - 1);
        
        part_3 += tmp*EPSILON + sqr(1-v[i])/(4*EPSILON);
    }
    part_3 *= BETA;
    
    
    if (flag==1) {
        AT = part_1+part_2+part_3;
        G = part_1+part_2;
        H = part_2+part_3;
    }
    printf("AT(f,v) = %f (%.2f%%),\tG(f) = %f (%.2f%%),\tH(v) = %f (%.2f%%)\n",part_1+part_2+part_3,(part_1+part_2+part_3)/AT*100.,
        part_1+part_2,(part_1+part_2)/G*100.,part_2+part_3,(part_2+part_3)/H*100.);
}

void min_wrapper(int np,int nr) {
    int line[IMGSIZE*2+10];
    float weight[IMGSIZE*2+10];
    int numb;
    float snorm;
    
    wray(np, nr, line, weight, &numb, &snorm);
    
    minIMAGE(np,nr,line,weight,numb,snorm);
    minEDGE(np,nr,line,weight,numb); 
}

void MSbeam() {
    int i,j,k;
    for (i = 0; i<IMGSIZE*IMGSIZE; ++i) {
        f[i] = 0.;
        v[i] = 1.;
    }
    
    lambda = 0.001;
        
    printf("Begin MSbeam minimization ...\n");
    for (i = 1; i<=ALL_ITER; ++i) {
        printf("Iteration %d ...\n", i);
        printf("lambda = %f\n",lambda);
        #pragma omp parallel for private(j,k)
        for (j = 0; j<NPROJ; ++j)
            for (k = 0; k<NRAY; ++k) 
                min_wrapper(j,k);
        lambda = lambda/(1 + 500 * lambda);
    }
    printf("MSbeam minimization done.\n");
}
