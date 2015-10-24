#ifndef MSBEAM_H__
#define MSBEAM_H__

class MSBeamBase {
public: 
  virtual void msbeam(float *f, float *v, float *g, int num_thread) {};
};

#endif