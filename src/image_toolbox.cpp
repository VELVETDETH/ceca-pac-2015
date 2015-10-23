
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath> // for some math operations

#include "image_toolbox.h"

// Helper functions. We do not need to expose these. Those special characters 
// will be named and represented in LaTeX conventions.
// For example: mu_x

double avg(double *x, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i];
    return (sum / n);
}

double var(double *x, int n) {
    double x_avg = avg(x, n);
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double dx = x[i] - x_avg;
        sum += dx * dx;
    }
    return (sum / n);
}

double covar(double *x, double *y, int n) {
    double x_avg = avg(x, n);
    double y_avg = avg(y, n);
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += (x[i] - x_avg) * (y[i] - y_avg);
    return (sum / n);
}

// returen the gaussian filter matrix
// The matrix should be N X N
double* gaussian_filter(int N, double sigma) {
    int filter_size = N * N;
    double *filter = (double *) malloc(sizeof(double) * filter_size);
    double sum = 0.0; // for normalization

    for (int i = 0; i < N; i ++) {
        for (int j = 0; j < N; j ++) {
            int idx = i * N + j;
            int x = i - N / 2;
            int y = j - N / 2;

            double sigma_sq = sigma * sigma;
            filter[idx]  = exp(-(x * x + y * y) / (2 * sigma_sq));
            filter[idx] *= 1.0 / (2 * M_PI * sigma_sq);
            sum += filter[idx];
        }
    }
    for (int i = 0; i < N; i ++) {
        for (int j = 0; j < N; j ++) {
            int idx = i * N + j;
            filter[idx] /= sum;
        }
    }
    return filter;
}

namespace ImageToolbox {

    // L is the dynamic range of image's pixel value range. As we're working 
    // on gray scale images, [0, 255] is a reasonable range.
    // For more information, please take a look at 
    //  https://en.wikipedia.org/wiki/Dynamic_range
    const double L = 255;

    // by default, k_1 = 0.01 and k_2 = 0.03(SSIM specification);
    const double k_1 = 0.01;
    const double k_2 = 0.03;

    // constants for gaussian filter used.
    const int SSIM_win_width = 11;
    // half window width
    const int SSIM_win_hwidth = SSIM_win_width >> 1;
    const int SSIM_win_size = SSIM_win_width * SSIM_win_width;
    
    const double gf_sigma = 1.5;

    // There's no check of x and y indices range in this function.
    double SSIM(int x, int y, float *I, float *J, int m, int n, double *filter) {
        // weighted window of image I
        double *w_I = (double *) malloc(sizeof(double) * SSIM_win_size);
        double *w_J = (double *) malloc(sizeof(double) * SSIM_win_size);
        // initialize
        for (int i = 0; i < SSIM_win_width; i++) {
            for (int j = 0; j < SSIM_win_width; j++) {
                int _x = x + i - SSIM_win_hwidth;
                int _y = y + j - SSIM_win_hwidth;
                
                int idx = _x * n + _y; // 
                int f_idx = i * SSIM_win_width + j;
                
                w_I[f_idx] = filter[f_idx] * I[idx];
                w_J[f_idx] = filter[f_idx] * J[idx];
            }
        }
        
        // calculate averages
        double mu_x = avg(w_I, SSIM_win_size);
        double mu_y = avg(w_J, SSIM_win_size);

        // calculate variances
        double sigma_x = var(w_I, SSIM_win_size);
        double sigma_y = var(w_J, SSIM_win_size);
        double sigma_xy = covar(w_I, w_J, SSIM_win_size);

        // c1 and c2
        double c_1 = (k_1 * L) * (k_1 * L); // dummy style
        double c_2 = (k_2 * L) * (k_2 * L);
        // final result
        double result = 
            ((2 * mu_x * mu_y + c_1) * (2 * sigma_xy + c_2)) /
            ((mu_x * mu_x + mu_y * mu_y + c_1) * 
             (sigma_x * sigma_x + sigma_y * sigma_y + c_2));
        return result;
    }

    double MSSIM(float *I, float *J, int m, int n) {
        double sum = 0.0;
        int num_pixel = 0;

        double *w = gaussian_filter(SSIM_win_width, gf_sigma);
        // iterate over all the available pixels
        for (int i = SSIM_win_hwidth; i < m - SSIM_win_hwidth; i ++) {
            for (int j = SSIM_win_hwidth; j < n - SSIM_win_hwidth; j ++) {
                double ssim = SSIM(i, j, I, J, m, n, w);
                sum += ssim;
                num_pixel ++; // just use counter, hate formula
            }
        }
        return sum / num_pixel;
    }
}