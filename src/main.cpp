#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <getopt.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

// self defined headers
#include "utility.h"
#include "proj.h"
// This is the definition of all msbeam related class
#include "msbeam.h"
#include "msbeam_cpu.h"
#include "msbeam_offload_cpu.h"
#include "msbeam_mic.h"
#include "image_toolbox.h"

#define DEFAULT_NUM_THREADS 8

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

int main(int argc,char **argv) {

    int nthreads;
    char c;
    char *data_file_name = NULL;
    MSBeamBase *MSBeam = new MSBeamCpu;
    string platform_name = "";

    while ((c = getopt(argc, argv, "t:f:m:")) != -1) {
        switch (c) {
            case 't':
                nthreads = atoi(optarg);
                break;
            case 'f':
                data_file_name = optarg;
                break;
            case 'm': 
                if (!strcmp(optarg, "cpu")) {
                    platform_name = string("cpu");
                    MSBeam = new MSBeamCpu;
                    break;
                } else if (!strcmp(optarg, "offload_cpu")) {
                    platform_name = string("offload_cpu");
                    MSBeam = new MSBeamOffloadCpu;
                    break;
                } else if (!strcmp(optarg, "mic")) {
                    platform_name = string("mic");
                    MSBeam = new MSBeamMic;
                    break;
                }else {
                    fprintf(stderr, "Unrecognized version of MSBeam: %s\n", 
                        optarg);
                    exit(1);
                }
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
    float *g = (float *) malloc(sizeof(float) * (NPROJ * NRAY));
    float *f = (float *) malloc(sizeof(float) * (IMGSIZE * IMGSIZE));
    float *v = (float *) malloc(sizeof(float) * (IMGSIZE * IMGSIZE));
    float *r = (float *) malloc(sizeof(float) * (IMGSIZE * IMGSIZE));

    printf("NUMBER OF THREADS=%d\n",nthreads);

    printf("Running read_phantom ...\n");
    read_phantom(f, data_file_name);
    read_phantom(r, data_file_name); // just don't want to use memcpy
    
    normalize(f);
    normalize(r);
    
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

    printf("Ready to calculate ...\n");

    // s_wtime: start wall time
    // e_wtime: end wall time
    double s_wtime = omp_get_wtime();
    MSBeam->msbeam(f, v, g, nthreads);
    double e_wtime = omp_get_wtime();

    double elapsed = (e_wtime - s_wtime)/(IMGSIZE * IMGSIZE * ALL_ITER)*1e6;
    printf("\033[0;32mOK!\033[0;m\n");
    printf("Wall time elapsed %.3lf us\n", elapsed);

    normalize(f);

    printf("Start to verify the result: \n");
    double mssim = ImageToolbox::MSSIM(r, f, IMGSIZE, IMGSIZE);
    printf("MSSIM: %.6lf\n", mssim);
    
    write_file(f, "img.dat");
    write_file(v, "edge.dat");

    // Out put the result images
    uint8_t *f_std = (uint8_t *) malloc(sizeof(uint8_t) * IMGSIZE * IMGSIZE);
    uint8_t *r_std = (uint8_t *) malloc(sizeof(uint8_t) * IMGSIZE * IMGSIZE);
    for (int i = 0; i < IMGSIZE * IMGSIZE; i++) {
        f_std[i] = (uint8_t) (f[i]);
        r_std[i] = (uint8_t) (r[i]);
    }

    stringstream img_fname_ss;
    img_fname_ss << platform_name << "f_image_" << nthreads << ".png";
    cout << "image will be written at " << img_fname_ss.str() << endl;
    Mat f_img(IMGSIZE, IMGSIZE, CV_8UC1, f_std);
    Mat r_img(IMGSIZE, IMGSIZE, CV_8UC1, r_std);
    imwrite(img_fname_ss.str(), f_img);

    stringstream res_fname_ss;
    res_fname_ss << platform_name <<  "msbeam_result.txt";
    ofstream result_file(res_fname_ss.str().c_str(), ios::out | ios::app);
    if (result_file.is_open()) {
        result_file << elapsed << ", " << mssim << endl;
    } else {
        cerr << "can't write file " << res_fname_ss.str() << endl;
    }

    return 0;
}
