
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "wray.h"
#include "proj.h"
#include "proto.h"
#include "utility.h"
#include "msbeam_offload_cpu.h"

namespace MSBeamOffloadCpuHelper {
    /**
     * In order to build this offload version, I've made following revision:
     * 1. Abandoned the hand-written sqr() function, with simple macro. But 
     * you should know that this will slow down the compiler a lot.
     * 2. Moved wray to an individual file.
     */

    __attribute__((target(mic)))
    void min_image(float *f, float *v, float *g, int np, int nr, int *line, 
        float *weight, int numb, float snorm, float lambda) {
        
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

    __attribute__((target(mic)))
    void min_edge(float *f, float *v, int np, int nr, int *line, float *weight,
        int numb, float lambda) {
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

    __attribute__((target(mic)))
    void min_wrapper(float *f, float *v, float *g, int np, int nr, 
        int *line, float *weight, float lambda) {
       
        memset(line, 0, sizeof(int) * IMGSIZE * 2);
        memset(weight, 0, sizeof(float) * IMGSIZE * 2);

        int numb;
        float snorm;
        
        wray(np, nr, line, weight, &numb, &snorm);
        
        min_image(f, v, g, np, nr, line, weight, numb, snorm, lambda);
        min_edge(f, v, np, nr, line, weight, numb, lambda); 
    }
}

void MSBeamOffloadCpu::msbeam(float *f, float *v, float *g, int num_thread) {
    int i,j,k;
    for (i = 0; i<IMGSIZE*IMGSIZE; ++i) {
        f[i] = 0.;
        v[i] = 1.;
    }
    printf("Image and edge data has been initialized.\n");
    
    printf("Begin msbeam offload ...\n");

    int f_size = IMGSIZE * IMGSIZE;
    int g_size = NPROJ * NRAY;

    #pragma offload target(mic) \
        inout(f[0:f_size]) \
        inout(g[0:g_size]) \
        inout(v[0:f_size])
    {
        printf("On MIC...\n"); fflush(0);
        float lambda = 0.001;

        int   **ls = (int**)   malloc(sizeof(int*)   * num_thread);
        float **ws = (float**) malloc(sizeof(float*) * num_thread);
        for (int i = 0; i < num_thread; i++) {
            ls[i] = (int*)   malloc(sizeof(int)   * IMGSIZE * 2);
            ws[i] = (float*) malloc(sizeof(float) * IMGSIZE * 2);
        }

        double s_wtime = omp_get_wtime();
        for (int i = 1; i <= ALL_ITER; ++i) {
            printf("Iteration %d lambda: %.6f\n", i, lambda);
            fflush(0);
            #pragma omp parallel for private(j) num_threads(num_thread)
            for (j = 0; j < NPROJ * NRAY; ++j) {
                int tid = omp_get_thread_num();
                int _j = j / NRAY;
                int _k = j % NRAY;
                MSBeamOffloadCpuHelper::min_wrapper(f, v, g, _j, _k, ls[tid], ws[tid], lambda);
            }
            
            lambda = lambda/(1 + 500 * lambda);
            fflush(0);
        }
        double e_wtime = omp_get_wtime();
        printf("Calculation wall time on MIC for each element: %.3lf us\n", 
            (e_wtime - s_wtime)/(f_size * ALL_ITER) * 1e6);
        fflush(0);

        for (int i = 0; i < num_thread; i++) {
            free(ls[i]);
            free(ws[i]);
        }
        free(ls);
        free(ws);
    }


    printf("msbeam minimization done.\n");
}
