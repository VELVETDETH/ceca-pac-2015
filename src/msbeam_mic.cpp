
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <omp.h>

#include "proto.h"
#include "proj.h"
#include "wray.h"
#include "utility.h"
#include "msbeam_mic.h"

namespace MSBeamMicHelper {
    __attribute__((target(mic)))
    void pre_cal(float *_a, float *_b, float *_c, float *_tmp, float *_lap, 
        float *f, float *v, float *g) {
        int idx;

        #pragma simd
        for(idx=0; idx<IMGSIZE*IMGSIZE; ++idx){
            int x = idx/IMGSIZE, y = idx%IMGSIZE;
            float tmp = 0.;
            float lap = 0.;

            float _v = v[idx];
            float _f = f[idx];
            float _f0= f[idx+IMGSIZE];
            float _f1= f[idx-IMGSIZE];
            float _f2= f[idx+1];
            float _f3= f[idx-1];
            float _v0= v[idx-IMGSIZE];
            float _v1= v[idx-1];

            if (x+1<IMGSIZE) tmp += sqr(_v)*(_f0-_f);
            else             tmp += sqr(_v)*(  0-_f);
            
            if (y+1<IMGSIZE) tmp += sqr(_v)*(_f2-_f);
            else             tmp += sqr(_v)*(  0-_f);
            
            if (x-1>=0)      tmp -= sqr(_v0)*(_f-_f1);
            else             tmp -= _f;
            
            if (y-1>=0)      tmp -= sqr(_v1)*(_f-_f3);
            else             tmp -= _f;
            
            if (x+1<IMGSIZE) lap += _f0;
            if (y+1<IMGSIZE) lap += _f2;
            if (x-1>=0)      lap += _f1;
            if (y-1>=0)      lap += _f3;
            lap -= 4*_f;
            
            _tmp[idx] = tmp;
            _lap[idx] = lap;
        }
         
        #pragma simd   
        for (idx = 0; idx<IMGSIZE*IMGSIZE; ++idx) {
            int x = idx/IMGSIZE, y = idx%IMGSIZE;
            
            float a = 0.;
            float b = 0.;
            float c = 0.;

            float _v = v[idx];
            float _f = f[idx];
            float _f0= f[idx-IMGSIZE];
            float _f1= f[idx-1];
            float _v0= v[idx+IMGSIZE];
            float _v1= v[idx+1];
            float _v2= v[idx-IMGSIZE];
            float _v3= v[idx-1];

            if (x-1>=0)      a += sqr(_f-_f0);
            else             a += sqr(_f-  0);
            
            if (y-1>=0)      a += sqr(_f-_f1);
            else             a += sqr(_f-  0);
            
            a *= _v;
            
            b = _v-1;
                    
            if (x+1<IMGSIZE) c += _v0;
            if (y+1<IMGSIZE) c += _v1;
            if (x-1>=0)      c += _v2;
            if (y-1>=0)      c += _v3;
            c -= (_v * 4);
            
            _a[idx] = a;
            _b[idx] = b;
            _c[idx] = c;
        }
    }

    __attribute__((target(mic)))
    void minIMAGE(int np, int *line, float *weight, int *nline, int *numb,
        float *tmp, float *lap, float *f, float *v, float *g, float lambda){
        int x, y;
        int idx;
        int whichline;
        float _Af;
        float d;
        float Af[NRAY]={0.0};
        for(x=0; x<IMGSIZE; ++x){
            for(y=0; y<numb[x]; ++y){
                int _idx = x*MAXPIX+y;

                idx = line[_idx];
                whichline = nline[_idx];
                Af[whichline] += f[idx]*weight[_idx];
            }
        }
        int nr;
        for(nr = 0; nr < NRAY; ++nr){
            Af[nr] -= g[np*NRAY+nr];
        }
        
        for(x = 0; x < IMGSIZE; ++x){
            for(y = 0; y < numb[x]; ++y){
                int _idx = x*MAXPIX+y;

                idx = line[_idx];
                whichline = nline[_idx];
                _Af = Af[whichline];
                d = -_Af * weight[_idx] + ALPHA * 
                    (tmp[idx] + sqr(EPSILON) * lap[idx]);
                f[idx] += lambda*d;
                
                if (f[idx]<0.0) f[idx] = 0.0;
                if (f[idx]>255.0) f[idx] = 255.0;
            }
        }
    }

    __attribute__((target(mic)))
    void minEDGE(int np,int *line,float *weight,int *numb, float*a, 
        float*b, float*c, float *f, float *v, float *g, float lambda) {
        int x,y;
        int idx;
        float d;
        for(x = 0; x < IMGSIZE; ++x){
            for(y = 0; y < numb[x]; ++y){
                idx = line[x*MAXPIX+y];
                d = -ALPHA*a[idx]-BETA/(4*EPSILON)*b[idx]+BETA*EPSILON*c[idx];
                v[idx] += lambda*d;
                if (v[idx]<0.0) v[idx] = 0.0;
                if (v[idx]>1.0) v[idx] = 1.0;
            }
        }
    }

    __attribute__((target(mic)))
    void min_wrapper(int np, float*a, float*b, float*c, float*tmp, float*lap, 
        float *f, float *v, float *g, int *line, float *weight, int *nline,
        float lambda) {
        int *numb = (int *) malloc(sizeof(int) * IMGSIZE * IMGSIZE);
        memset(numb, 0, sizeof(int) * IMGSIZE * IMGSIZE);
        
        int l_size = IMGSIZE * NRAY * 3;
        
        memset(line,   0, l_size * sizeof(int));
        memset(weight, 0, l_size * sizeof(float));
        memset(nline,  0, l_size * sizeof(int));

        int nr;
        for(nr = 0; nr < NRAY; ++ nr){
            wray_new(np, nr, line, weight, nline, numb, f);
        }
        
        minIMAGE(np,line,weight,nline,numb,tmp,lap,f,v,g,lambda);
        minEDGE(np,line,weight,numb,a,b,c,f,v,g,lambda);
    }
};

void MSBeamMic::msbeam(float *f, float *v, float *g, int num_thread) {
    int i,j,k;
    int l_size = IMGSIZE * NRAY * 3;
    int f_size = IMGSIZE * IMGSIZE;
    int g_size = NPROJ * NRAY;

    float *a   = (float *) malloc(sizeof(float) * f_size);
    float *b   = (float *) malloc(sizeof(float) * f_size);
    float *c   = (float *) malloc(sizeof(float) * f_size);
    float *tmp = (float *) malloc(sizeof(float) * f_size);
    float *lap = (float *) malloc(sizeof(float) * f_size);

    memset(a,   0, sizeof(float) * f_size);
    memset(b,   0, sizeof(float) * f_size);
    memset(c,   0, sizeof(float) * f_size);
    memset(tmp, 0, sizeof(float) * f_size);
    memset(lap, 0, sizeof(float) * f_size);
    
    

    for (i = 0; i<IMGSIZE*IMGSIZE; ++i) {
        f[i] = 0.;
        v[i] = 0.;
    }

    #pragma offload target(mic) \
        in(f:      length(f_size) alloc_if(1) free_if(0)) \
        in(g:      length(g_size) alloc_if(1) free_if(0)) \
        in(v:      length(f_size) alloc_if(1) free_if(0)) \
        in(a:      length(f_size) alloc_if(1) free_if(0)) \
        in(b:      length(f_size) alloc_if(1) free_if(0)) \
        in(c:      length(f_size) alloc_if(1) free_if(0)) \
        in(tmp:    length(f_size) alloc_if(1) free_if(0)) \
        in(lap:    length(f_size) alloc_if(1) free_if(0)) 
    {}

    
    #pragma offload target(mic) \
        nocopy(f[0:f_size]) \
        nocopy(g[0:g_size]) \
        nocopy(v[0:f_size]) \
        nocopy(a[0:f_size]) \
        nocopy(b[0:f_size]) \
        nocopy(c[0:f_size]) \
        nocopy(tmp[0:f_size]) \
        nocopy(lap[0:f_size]) 
    {
        float lambda = 0.007;
        // initialize helper arrays
        int   **ls = (int**)   malloc(sizeof(int*)   * num_thread);
        int   **ns = (int**)   malloc(sizeof(int*)   * num_thread);
        float **ws = (float**) malloc(sizeof(float*) * num_thread);
        for (int i = 0; i < num_thread; i++) {
            ls[i] = (int*)   malloc(sizeof(int)   * l_size);
            ns[i] = (int*)   malloc(sizeof(int)   * l_size);
            ws[i] = (float*) malloc(sizeof(float) * l_size);
        }

        double s_wtime = omp_get_wtime();
        for (int i = 1; i <= ALL_ITER; ++i) {
            printf("Iteration %2d lambda: %.6f\n", i, lambda); fflush(0);
            MSBeamMicHelper::pre_cal(a,b,c,tmp,lap,f,v,g);
            #pragma omp parallel for private(j) num_threads(num_thread)
            for (j = 0; j<NPROJ; ++j) {
                int tid = omp_get_thread_num();
                MSBeamMicHelper::min_wrapper(j,a,b,c,tmp,lap,f,v,g,ls[tid],ws[tid],ns[tid],lambda);
            }
            
            lambda = lambda/(1 + 2500 * lambda);
        }
        double e_wtime = omp_get_wtime();
        printf("Calculation wall time on MIC for each element: %.3lf us\n", 
            (e_wtime - s_wtime)/(f_size * ALL_ITER) * 1e6);
        fflush(0);

        for (int i = 0; i < num_thread; i++) {
            free(ls[i]);
            free(ns[i]);
            free(ws[i]);
        }
        free(ls);
        free(ns);
        free(ws);
    }
    
    printf("\033[0;32mOK!\033[0;m\n");
    

    #pragma offload target(mic) \
        out(f: length(f_size) alloc_if(0) free_if(1)) \
        out(g: length(g_size) alloc_if(0) free_if(1)) \
        out(v: length(f_size) alloc_if(0) free_if(1)) 
    {}
    
    printf("MSbeam minimization done.\n");
}