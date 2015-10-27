
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
            if (x+1<IMGSIZE) tmp += sqr(v[idx])*(f[idx+IMGSIZE]-f[idx]);
            else             tmp += sqr(v[idx])*(   0          -f[idx]);
            
            if (y+1<IMGSIZE) tmp += sqr(v[idx])*(f[idx+1]-f[idx]);
            else             tmp += sqr(v[idx])*(   0    -f[idx]);
            
            if (x-1>=0)      tmp -= sqr(v[idx-IMGSIZE])*(f[idx]-f[idx-IMGSIZE]);
            else             tmp -=                     (f[idx]-0        );
            
            if (y-1>=0)      tmp -= sqr(v[idx-1])*(f[idx]-f[idx-1]);
            else             tmp -=               (f[idx]-0       );
            
            if (x+1<IMGSIZE) lap += f[idx+IMGSIZE];
            if (y+1<IMGSIZE) lap += f[idx+1];
            if (x-1>=0)      lap += f[idx-IMGSIZE];
            if (y-1>=0)      lap += f[idx-1];
            lap -= 4*f[idx];
            
            _tmp[idx] = tmp;
            _lap[idx] = lap;
        }
         
        #pragma simd   
        for (idx = 0; idx<IMGSIZE*IMGSIZE; ++idx) {
            int x = idx/IMGSIZE, y = idx%IMGSIZE;
            
            float a = 0.;
            float b = 0.;
            float c = 0.;

            if (x-1>=0)      a += sqr(f[idx]-f[idx-IMGSIZE]);
            else             a += sqr(f[idx]-0        );
            
            if (y-1>=0)      a += sqr(f[idx]-f[idx-1]);
            else             a += sqr(f[idx]-0       );
            
            a *= v[idx];
            
            b = v[idx]-1;
                    
            if (x+1<IMGSIZE) c += v[idx+IMGSIZE];
            if (y+1<IMGSIZE) c += v[idx+1];
            if (x-1>=0)      c += v[idx-IMGSIZE];
            if (y-1>=0)      c += v[idx-1];
            c -= 4*v[idx];
            
            _a[idx] = a;
            _b[idx] = b;
            _c[idx] = c;
        }
    }

    __attribute__((target(mic)))
    void minIMAGE(int np, int *line, float *weight, int *nline, int numb[IMGSIZE],
        float *tmp, float *lap, float *f, float *v, float *g, float lambda){
        int x, y;
        int idx;
        int whichline;
        float _Af;
        float d;
        float Af[NRAY]={0.0};
        for(x=0; x<IMGSIZE; ++x){
            for(y=0; y<numb[x]; ++y){
                idx = line[x*MAXPIX+y];
                whichline = nline[x*MAXPIX+y];
                Af[whichline] += f[idx]*weight[x*MAXPIX+y];
            }
        }
        int nr;
        for(nr = 0; nr < NRAY; ++nr){
            Af[nr] -= g[np*NRAY+nr];
        }
        
        for(x = 0; x < IMGSIZE; ++x){
            for(y = 0; y < numb[x]; ++y){
                idx = line[x*MAXPIX+y];
                whichline = nline[x*MAXPIX+y];
                _Af = Af[whichline];
                d = -_Af * weight[x*MAXPIX+y] + ALPHA * 
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
        int numb[IMGSIZE]={0};
        
        int l_size = 0;
        memset(line, 0, l_size * sizeof(int));
        memset(weight, 0, l_size * sizeof(float));
        memset(nline, 0, l_size * sizeof(int));

        int nr;
        for(nr=0;nr<NRAY;++nr){
            wray_new(np, nr, line, weight,nline,numb,f);
        }
        
        minIMAGE(np,line,weight,nline,numb,tmp,lap,f,v,g,lambda);
        minEDGE(np,line,weight,numb,a,b,c,f,v,g,lambda);
    }
}

void MSBeamMic::msbeam(float *f, float *v, float *g, int num_thread) {
    int i,j,k;
    int l_size = IMGSIZE * NRAY * 3;
    int f_size = IMGSIZE * IMGSIZE;
    int g_size = NPROJ * NRAY;

    float* a = (float *) malloc(sizeof (float) * f_size);
    float* b = (float *) malloc(sizeof (float) * f_size);
    float* c = (float *) malloc(sizeof (float) * f_size);
    float* tmp = (float *) malloc(sizeof (float) * f_size);
    float* lap = (float *) malloc(sizeof (float) * f_size);
    
    int *line = (int*) malloc(sizeof(int)*IMGSIZE*NRAY*3);
    float *weight = (float*) malloc(sizeof(float)*IMGSIZE*NRAY*3);
    int *nline = (int*) malloc(sizeof(int)*IMGSIZE*NRAY*3);

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
        in(lap:    length(f_size) alloc_if(1) free_if(0)) \
        in(line:   length(l_size) alloc_if(1) free_if(0)) \
        in(weight: length(l_size) alloc_if(1) free_if(0)) \
        in(nline:  length(l_size) alloc_if(1) free_if(0))
    {}

    double s_wtime = omp_get_wtime();
    #pragma offload target(mic) \
        nocopy(f[0:f_size]) \
        nocopy(g[0:g_size]) \
        nocopy(v[0:f_size]) \
        nocopy(a[0:f_size]) \
        nocopy(b[0:f_size]) \
        nocopy(c[0:f_size]) \
        nocopy(tmp[0:f_size]) \
        nocopy(lap[0:f_size]) \
        nocopy(line[0:l_size]) \
        nocopy(weight[0:l_size]) \
        nocopy(nline[0:l_size])
    {
        float lambda = 0.007;
        for (i = 1; i <= ALL_ITER; ++i) {
            
            MSBeamMicHelper::pre_cal(a,b,c,tmp,lap,f,v,g);
            #pragma omp parallel for private(j) num_threads(num_thread)
            for (j = 0; j<NPROJ; ++j) {
                MSBeamMicHelper::min_wrapper(j,a,b,c,tmp,lap,f,v,g,line,weight,nline,lambda);
            }
            
            lambda = lambda/(1 + 2500 * lambda);
        }
    }
    double e_wtime = omp_get_wtime();
    printf("\033[0;32mOK!\033[0;m\n");
    printf("Calculation wall time elapsed %.3lf s\n", e_wtime - s_wtime);
    printf("Calculation wall time for each element: %.3lf us\n", 
        (e_wtime - s_wtime)/(f_size) * 1e6);

    #pragma offload target(mic) \
        out(f: length(f_size) alloc_if(0) free_if(1)) \
        out(g: length(g_size) alloc_if(0) free_if(1)) \
        out(v: length(f_size) alloc_if(0) free_if(1)) 
    {}

    free(line);
    free(weight);
    free(nline);
    
    printf("MSbeam minimization done.\n");
}