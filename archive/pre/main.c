#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <immintrin.h>
#include "proj.h"
#include "proto.h"
#include "utility.h"
#define NT 240
float g[NPROJ*NRAY+10];

float f[IMGSIZE*IMGSIZE+10];
float v[IMGSIZE*IMGSIZE+10];
int testIter;
float lambda;
int outcount=0;
void MSbeam();
long altime[NPROJ][NRAY];
long t1;
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
  // init the timer for real and proc(what?)
  init(&timer_main_real, CLOCK_REALTIME);
	init(&timer_main_proc, CLOCK_PROCESS_CPUTIME_ID);
  // why we need to do a tick here?
	tick(&timer_main_real);
	tick(&timer_main_proc);
  int nthreads;

  // here simply initialize several const variables
  // ini_const_m512();

  // command line process: Only nthread variable
  if (argc>1) {
    nthreads = 0;
    // what... just use atoi
    int len = strlen(argv[1]),k;
    for (k = 0; k<len; ++k) 
      nthreads = nthreads*10+argv[1][k]-'0';
  }
  else 
    nthreads = 240;
  
  printf("NUMBER OF THREADS=%d\n",nthreads);
  omp_set_num_threads(nthreads);

  /**
   * read_phantom (utility.c): 
   * Initialize the data in f array. According to the paper, f is 
   * representing the image. OK, I've seen the code of read_phantom, it's
   * totally useless as it's simply read the .dat file in...
   * The only thing we need to take care of is that the image size has been 
   * fixed(IMGSIZE=512)
   */
  read_phantom(f, "phant_g1.dat");

  /* normalize (utility.c): Just put the data range in [0,255] */
  normalize(f);

  /* write_file (utility.c): write the image back to file system */
  write_file(f, "std.dat");
  
  /**
   * A (proj.c): 
   * Takes 2 input array, f is the image and g is the X-ray data to
   * be processed. It's size is NPROJ X NRAY, where NPROJ equals to 768
   * and NRAY is 180, just like the specification in the paper(is it equal to
   * the real-world machine?)
   */
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

  printf("TOT TIME USED: %.2f\n",(float)(end-start));
  
  normalize(f);
  
  write_file(f, "img.dat");
  
  write_file(v, "edge.dat");
  
  return 0;
}

void mminIMAGE(int np, int nr, __m512i *line, __m512 *weight, __m512i numb, float snorm) {

  int i;
  __m512 d[IMGSIZE*2+10];
  int maxnumb = _mm512_reduce_max_epi32(numb);
  zmmf_t Af;
  Af.reg = _mm512_set1_ps(0.0);
  for (i = 0; i<maxnumb; ++i) {
    __m512i ind = line[i];
    Af.reg = _mm512_fmadd_ps(_mm512_i32gather_ps(ind,f,4),weight[i],Af.reg);
  }
  for(i = 0; i<16; ++i) {
    Af.elems[i] -= g[np*NRAY+nr+i];
  }
    
  for (i = 0; i<maxnumb; ++i) {
    __m512i ind = line[i];
  	__m512i ind_m_1, ind_m_IMG,ind_p_1,ind_p_IMG;
    __m512i x = _mm512_div_epi32(ind,_mm512_set1_epi32(IMGSIZE));
  	
  	__m512i y = _mm512_rem_epi32(ind,_mm512_set1_epi32(IMGSIZE));

    __m512 tmp = _mm512_set1_ps(0.0);
    __m512 lap = _mm512_set1_ps(0.0);
    __mmask16 J1,J2,J3,J4,rJ1,rJ2,rJ3,rJ4;
  	__m512 m1j1,m2j1,m1rj1,m2rj1,m1j2,m2j2,m1rj2,m2rj2,m1j3,m2j3,m1rj3,m2rj3,
  		m1j4,m2j4,m1rj4,m2rj4;
  	__m512 f_p_IMG,f_m_IMG,f_m_1,f_p_1,f_0;
  	__m512 v_m_IMG,v_0,v_m_1;
  	J1  = _mm512_cmplt_epi32_mask(_mm512_add_epi32(x,_mm512_set1_epi32(1)),mIMGSIZE);
  	rJ1 = _mm512_cmpge_epi32_mask(_mm512_add_epi32(x,_mm512_set1_epi32(1)),mIMGSIZE);
  	J2  = _mm512_cmplt_epi32_mask(_mm512_add_epi32(y,_mm512_set1_epi32(1)),mIMGSIZE);
  	rJ2 = _mm512_cmpge_epi32_mask(_mm512_add_epi32(y,_mm512_set1_epi32(1)),mIMGSIZE);
  	J3  = _mm512_cmpge_epi32_mask(x,_mm512_set1_epi32(1));
  	rJ3 = _mm512_cmplt_epi32_mask(x,_mm512_set1_epi32(1));
  	J4  = _mm512_cmpge_epi32_mask(y,_mm512_set1_epi32(1));
  	rJ4 = _mm512_cmplt_epi32_mask(y,_mm512_set1_epi32(1));

  	ind_m_1 = _mm512_sub_epi32(ind, _mm512_set1_epi32(1) );
  	ind_m_IMG = _mm512_sub_epi32(ind, _mm512_set1_epi32(IMGSIZE) );
  	ind_p_1 = _mm512_add_epi32(ind, _mm512_set1_epi32(1) );
  	ind_p_IMG = _mm512_add_epi32(ind, numiIMG );

  	f_p_IMG = _mm512_mask_i32gather_ps(_mm512_set1_ps(0.0),_mm512_cmplt_epi32_mask(ind_p_IMG,numiIMG2),ind_p_IMG,f,4);
  	f_m_IMG = _mm512_mask_i32gather_ps(_mm512_set1_ps(0.0),_mm512_cmpge_epi32_mask(ind_m_IMG,_mm512_set1_epi32(0)),ind_m_IMG,f,4);
  	f_p_1   = _mm512_mask_i32gather_ps(_mm512_set1_ps(0.0),_mm512_cmplt_epi32_mask(ind_p_1,numiIMG2),ind_p_1,f,4);
  	f_m_1   = _mm512_mask_i32gather_ps(_mm512_set1_ps(0.0),_mm512_cmpge_epi32_mask(ind_m_1,_mm512_set1_epi32(0)),ind_m_1,f,4);
  	f_0     = _mm512_i32gather_ps(ind,f,4);

  	v_m_IMG = _mm512_mask_i32gather_ps(_mm512_set1_ps(0.0),_mm512_cmpge_epi32_mask(ind_m_IMG,_mm512_set1_epi32(0)),ind_m_IMG,v,4);
  	v_0     = _mm512_i32gather_ps(ind,v,4);
  	v_m_1   = _mm512_mask_i32gather_ps(_mm512_set1_ps(0.0),_mm512_cmpge_epi32_mask(ind_m_1,_mm512_set1_epi32(0)),ind_m_1,v,4);

  	m1j1 = _mm512_mul_ps(v_0,v_0);
  	m2j1 = _mm512_sub_ps(f_p_IMG,f_0);
  	m1rj1 = _mm512_mul_ps(v_0,v_0);
  	m2rj1 = _mm512_sub_ps(_mm512_set1_ps(0.0),f_0);

  	m1j2 = _mm512_mul_ps(v_0,v_0);
  	m2j2 = _mm512_sub_ps(f_p_1,f_0);
  	m1rj2 = _mm512_mul_ps(v_0,v_0);
  	m2rj2 = _mm512_sub_ps(_mm512_set1_ps(0.0),f_0);

  	m1j3 = _mm512_mul_ps(v_m_IMG,v_m_IMG);
  	m2j3 = _mm512_sub_ps(f_0,f_m_IMG);
  	m1rj3 = _mm512_set1_ps(1.0);	
  	m2rj3 = _mm512_add_ps(f_0,_mm512_set1_ps(0.0));

  	m1j4 = _mm512_mul_ps(v_m_1,v_m_1);
  	m2j4 = _mm512_sub_ps(f_0,f_m_1);
  	m1rj4 = _mm512_set1_ps(1.0);
  	m2rj4 = _mm512_add_ps(f_0,_mm512_set1_ps(0.0));
  		
  	tmp = _mm512_mask_add_ps(tmp,J1,tmp,_mm512_mul_ps(m1j1,m2j1));
  	tmp = _mm512_mask_add_ps(tmp,rJ1,tmp,_mm512_mul_ps(m1rj1,m2rj1));
  	tmp = _mm512_mask_add_ps(tmp,J2,tmp,_mm512_mul_ps(m1j2,m2j2));
  	tmp = _mm512_mask_add_ps(tmp,rJ2,tmp,_mm512_mul_ps(m1rj2,m2rj2));
  	tmp = _mm512_mask_sub_ps(tmp,J3,tmp,_mm512_mul_ps(m1j3,m2j3));
  	tmp = _mm512_mask_sub_ps(tmp,rJ3,tmp,_mm512_mul_ps(m1rj3,m2rj3));
  	tmp = _mm512_mask_sub_ps(tmp,J4,tmp,_mm512_mul_ps(m1j4,m2j4));
  	tmp = _mm512_mask_sub_ps(tmp,rJ4,tmp,_mm512_mul_ps(m1rj4,m2rj4));

  	lap = _mm512_mask_add_ps(lap,J1,lap,f_p_IMG);
  	lap = _mm512_mask_add_ps(lap,J2,lap,f_p_1);
    lap = _mm512_mask_add_ps(lap,J3,lap,f_m_IMG);
    lap = _mm512_mask_add_ps(lap,J4,lap,f_m_1);
  	lap = _mm512_sub_ps(lap,_mm512_mul_ps(numf4,f_0) );

  	d[i] = _mm512_fmadd_ps(_mm512_set1_ps(EPSILON*EPSILON),lap,tmp);
  	d[i] = _mm512_mul_ps(_mm512_set1_ps(ALPHA),d[i]);
  	d[i] = _mm512_sub_ps(d[i],_mm512_mul_ps(Af.reg,weight[i]));
  }

  for (i = 0; i<maxnumb; ++i) {
    __m512i ind = line[i];
  	__m512 tmpf_ind =  _mm512_i32gather_ps(ind,f,4);
  	__mmask16 overidx = _mm512_cmplt_epi32_mask(_mm512_set1_epi32(i),numb);

  	tmpf_ind = _mm512_fmadd_ps(_mm512_set1_ps(lambda),d[i],tmpf_ind);
  	tmpf_ind = _mm512_mask_blend_ps(_mm512_cmplt_ps_mask(tmpf_ind,_mm512_set1_ps(0.0)),tmpf_ind,_mm512_set1_ps(0.0));	
  	tmpf_ind = _mm512_mask_blend_ps(_mm512_cmplt_ps_mask(_mm512_set1_ps(255.0),tmpf_ind),tmpf_ind,_mm512_set1_ps(255.0));

  	_mm512_mask_i32scatter_ps(f,overidx,ind,tmpf_ind,4);
  }
}

void mminEDGE(int np,int nr,__m512i *line,__m512 *weight,__m512i numb) {
  int i;

  __m512 d[IMGSIZE*2+10];
  int maxnumb = _mm512_reduce_max_epi32(numb);

  for (i = 0; i<maxnumb; ++i) {
    __m512i ind = line[i];
  	__m512i ind_m_1, ind_m_IMG,ind_p_1,ind_p_IMG;
          __m512i x = _mm512_div_epi32(ind,mIMGSIZE);
  	__m512i y = _mm512_rem_epi32(ind,mIMGSIZE);
          
  	__m512 a = _mm512_set1_ps(0.0);
          __m512 b = _mm512_set1_ps(0.0);
  	__m512 c = _mm512_set1_ps(0.0);
  	
  	__m512 f_m_IMG,f_m_1,f_0;
  	__m512 v_p_IMG,v_m_IMG,v_0,v_p_1,v_m_1;
  	__mmask16 J1,rJ1,J2,rJ2,J3,J4;
  	
  	ind_m_1 = _mm512_sub_epi32(ind, _mm512_set1_epi32(1) );
  	ind_m_IMG = _mm512_sub_epi32(ind, _mm512_set1_epi32(IMGSIZE) );
  	ind_p_1 = _mm512_add_epi32(ind, _mm512_set1_epi32(1) );
  	ind_p_IMG = _mm512_add_epi32(ind, _mm512_set1_epi32(IMGSIZE) );


  	f_m_IMG = _mm512_mask_i32gather_ps(_mm512_set1_ps(0.0),_mm512_cmpge_epi32_mask(ind_m_IMG,_mm512_set1_epi32(0)), ind_m_IMG,f,4);
  	f_m_1   = _mm512_mask_i32gather_ps(_mm512_set1_ps(0.0),_mm512_cmpge_epi32_mask(ind_m_1,_mm512_set1_epi32(0)),ind_m_1,f,4);
  	f_0     = _mm512_i32gather_ps(ind,f,4);
  	v_0     = _mm512_i32gather_ps(ind,v,4);
  	v_p_1   = _mm512_mask_i32gather_ps(_mm512_set1_ps(0.0),_mm512_cmplt_epi32_mask(ind_p_1,numiIMG2),ind_p_1,v,4);
  	v_m_1   = _mm512_mask_i32gather_ps(_mm512_set1_ps(0.0),_mm512_cmpge_epi32_mask(ind_m_1,_mm512_set1_epi32(0)),ind_m_1,v,4);
  	v_p_IMG = _mm512_mask_i32gather_ps(_mm512_set1_ps(0.0),_mm512_cmplt_epi32_mask(ind_p_IMG,numiIMG2),ind_p_IMG,v,4);
  	v_m_IMG = _mm512_mask_i32gather_ps(_mm512_set1_ps(0.0),_mm512_cmpge_epi32_mask(ind_m_IMG,_mm512_set1_epi32(0)),ind_m_IMG,v,4);
  	
  	J1  = _mm512_cmpge_epi32_mask(x,_mm512_set1_epi32(1));
  	rJ1 = _mm512_cmplt_epi32_mask(x,_mm512_set1_epi32(1));
  	J2  = _mm512_cmpge_epi32_mask(y,_mm512_set1_epi32(1));
  	rJ2 = _mm512_cmplt_epi32_mask(y,_mm512_set1_epi32(1));
  	J3  = _mm512_cmplt_epi32_mask(_mm512_add_epi32(x,_mm512_set1_epi32(1)),mIMGSIZE);
  	J4  = _mm512_cmplt_epi32_mask(_mm512_add_epi32(y,_mm512_set1_epi32(1)),mIMGSIZE);

  	a = _mm512_mask_add_ps(a,J1,a,_mm512_mul_ps(_mm512_sub_ps(f_0,f_m_IMG),_mm512_sub_ps(f_0,f_m_IMG)));
  	a = _mm512_mask_add_ps(a,rJ1,a,_mm512_mul_ps(f_0,f_0));
  	a = _mm512_mask_add_ps(a,J2,a,_mm512_mul_ps(_mm512_sub_ps(f_0,f_m_1),_mm512_sub_ps(f_0,f_m_1))); 
    a = _mm512_mask_add_ps(a,rJ2,a,_mm512_mul_ps(f_0,f_0));

  	a = _mm512_mul_ps(a,v_0);

  	b = _mm512_sub_ps(v_0,_mm512_set1_ps(1.0));

  	c = _mm512_mask_add_ps(c,J3,c,v_p_IMG);
  	c = _mm512_mask_add_ps(c,J4,c,v_p_1);
  	c = _mm512_mask_add_ps(c,J1,c,v_m_IMG);
  	c = _mm512_mask_add_ps(c,J2,c,v_m_1);
  	c = _mm512_sub_ps(c,_mm512_mul_ps(_mm512_set1_ps(4.0),v_0) );
          

  	d[i] = _mm512_sub_ps(_mm512_mul_ps(_mm512_set1_ps(BETA*EPSILON),c),_mm512_mul_ps(_mm512_set1_ps(BETA/(4*EPSILON)),b));
  	d[i] = _mm512_sub_ps(d[i],_mm512_mul_ps(_mm512_set1_ps(ALPHA),a) );
  }


  for (i = 0; i<maxnumb; ++i) {
    __m512i ind = line[i];
  	__m512 tmpv_ind =  _mm512_i32gather_ps(ind,v,4);
  	__mmask16 overidx = _mm512_cmplt_epi32_mask(_mm512_set1_epi32(i),numb);

  	tmpv_ind = _mm512_fmadd_ps(_mm512_set1_ps(lambda),d[i],tmpv_ind);
  	tmpv_ind = _mm512_mask_blend_ps(_mm512_cmplt_ps_mask(tmpv_ind,_mm512_set1_ps(0.0)),tmpv_ind,_mm512_set1_ps(0.0));	
  	tmpv_ind = _mm512_mask_blend_ps(_mm512_cmplt_ps_mask(_mm512_set1_ps(1.0),tmpv_ind),tmpv_ind,_mm512_set1_ps(1.0));

  	_mm512_mask_i32scatter_ps(v,overidx,ind,tmpv_ind,4);
  }
}

float AT_g[NPROJ*NRAY];

float AT,G,H;

void compute_AT(int flag) {
    A(AT_g, f);
    
    int i;
    float part_1 = 0., part_2 = 0., part_3 = 0.;
//#pragma simd 
    for (i = 0; i<NPROJ*NRAY; ++i)
        part_1 += sqr(AT_g[i]-g[i]);
//#pragma simd 
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
//#pragma simd  
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

void min_wrapper(int np,int nr,int* al,float*aw,int allnumb[NPROJ][NRAY]) {
    float snorm;
    int tn;
    int maxnumb;
    __m512i midx=_mm512_set_epi32(30*IMGSIZE,28*IMGSIZE,26*IMGSIZE,24*IMGSIZE,22*IMGSIZE,20*IMGSIZE,18*IMGSIZE,16*IMGSIZE,14*IMGSIZE,12*IMGSIZE,10*IMGSIZE,8*IMGSIZE,6*IMGSIZE,4*IMGSIZE,2*IMGSIZE,0);
    __m512i mline[IMGSIZE*2+10];
    __m512 mweight[IMGSIZE*2+10];
    zmmi_t mnumb;
    for(tn=0;tn<16;++tn){ 
   	 mnumb.elems[tn] = allnumb[np][nr+tn];
    }
    maxnumb=_mm512_reduce_max_epi32(mnumb.reg);
    for(tn=0;tn<maxnumb;++tn){
	int i1 = np*NRAY*2*IMGSIZE;
	int i2 = nr*2*IMGSIZE+tn;
	mline[tn]=_mm512_i32gather_epi32(midx, al+i1+i2 ,4);
	mweight[tn]=_mm512_i32gather_ps(midx,aw+i1+i2 ,4);
    }

    mminIMAGE(np,nr,mline,mweight,mnumb.reg,snorm);
    mminEDGE(np,nr,mline,mweight,mnumb.reg);
 
}


void MSbeam() {
    int i,j,k;
    for (i = 0; i<IMGSIZE*IMGSIZE; ++i) {
        f[i] = 0.;
        v[i] = 1.;
    }
    
    lambda = 0.001;
        
    printf("Begin MSbeam minimization ...\n");
    int* allline=(int*)malloc(sizeof(int)*NPROJ*NRAY*2*IMGSIZE);
    float* allweight=(float*)malloc(sizeof(float)*NPROJ*NRAY*2*IMGSIZE);
    int allnumb[NPROJ][NRAY];

    for(j = 0; j<NPROJ; ++j){
#pragma omp parallel for private(k)
	for(k = 0; k<NRAY; ++k){
	     int i1=j*NRAY*IMGSIZE*2;
	     int i2=k*IMGSIZE*2;
       	     wray(j, k, allline+i1+i2,allweight+i1+i2, &allnumb[j][k]);
	}	
    }
    for (i = 1; i<=ALL_ITER; ++i) {
        printf("Iteration %d ...\n", i);
	printf("lambda = %f\n",lambda);
        for (j = 0; j<NPROJ; ++j){		
#pragma omp parallel for private(k,t1)
	   for (k = 0; k<NRAY; k+=16) {
		    min_wrapper(j,k,allline,allweight,allnumb);
	    }	   
	}
	   
        lambda = lambda/(1 + 500 * lambda);
    }
    printf("MSbeam minimization done.\n");
}
