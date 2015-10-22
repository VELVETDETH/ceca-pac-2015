#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <getopt.h>

// self defined headers
#include "proj.h"
#include "proto.h"
#include "utility.h"

#define DEFAULT_NUM_THREADS 8

float lambda;

void MSbeam(float *f, float *v, float *g);

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

int main(int argc,char **argv) {

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

    printf("Initializing memory ...\n");
    float *g = (float *) malloc(sizeof(float) * (NPROJ*NRAY));
    float *f = (float *) malloc(sizeof(float) * (IMGSIZE*IMGSIZE));
    float *v = (float *) malloc(sizeof(float) * (IMGSIZE*IMGSIZE));


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

    // s_wtime: start wall time
    // e_wtime: end wall time
    double s_wtime = omp_get_wtime();
    MSbeam(f, v, g);
    double e_wtime = omp_get_wtime();

    printf("\033[0;32mOK!\033[0;m\n");
    printf("Wall time elapsed %.3lf s\n", end_wtime - start_wtime);
    
    normalize(f);
    
    write_file(f, "img.dat");
    
    write_file(v, "edge.dat");
    
    return 0;
}


void minIMAGE(float *f, float *v, float *g, int np, int nr, int *line, 
    float *weight, int numb, float snorm) {
    
    int i;
    float d[IMGSIZE*2+10];
    float Af = 0.;

    for (i = 0; i<numb; ++i) {
        int ind = line[i];
        Af += f[ind]*weight[i];
    }
    Af -= g[np*NRAY+nr];
    
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

void minEDGE(float *f, float *v, int np,int nr,int *line,float *weight,int numb) {
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

void min_wrapper(float *f, float *v, float *g, int np,int nr) {
    int line[IMGSIZE*2+10];
    float weight[IMGSIZE*2+10];
    int numb;
    float snorm;
    
    wray(np, nr, line, weight, &numb, &snorm);
    
    minIMAGE(f, v, g, np, nr, line, weight, numb, snorm);
    minEDGE(f, v, np, nr, line, weight, numb); 
}

void MSbeam(float *f, float *v, float *g) {
    int i,j,k;
    for (i = 0; i<IMGSIZE*IMGSIZE; ++i) {
        f[i] = 0.;
        v[i] = 1.;
    }
    printf("Image and edge data has been initialized.\n");
    
    lambda = 0.001;
        
    printf("Begin MSbeam minimization ...\n");
    for (i = 1; i<=ALL_ITER; ++i) {
        printf("\033[0;31mIteration\033[0m %d ...\n", i);
        printf("lambda = %f\n",lambda);
        
        #pragma omp parallel for private(j,k)
        for (j = 0; j < NPROJ; ++j)
            for (k = 0; k < NRAY; ++k) 
                min_wrapper(f, v, g, j, k);
        
        lambda = lambda/(1 + 500 * lambda);
    }
    printf("MSbeam minimization done.\n");
}
