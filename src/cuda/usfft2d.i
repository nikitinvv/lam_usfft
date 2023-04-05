/*interface*/
%module usfft2d

%{
#define SWIG_FILE_WITH_INIT
#include "usfft2d.cuh"
%}

class usfft2d {

public:  
  %mutable;  
  usfft2d(size_t n0, size_t n1, size_t n2, size_t ntheta, size_t detw, size_t deth);
  ~usfft2d();  
  void fwd(size_t g_, size_t f_, size_t theta_, float phi, int k, int deth0, size_t stream_);
  void free();
};