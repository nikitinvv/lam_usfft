/*interface*/
%module fft

%{
#define SWIG_FILE_WITH_INIT
#include "fft.cuh"
%}

class fft {

public:
  %immutable;
  size_t n0,n1,n2,m;  
  
  %mutable;  
  fft(size_t n0, size_t n1, size_t n2, size_t m);
  ~fft();  
  void fwd1d(size_t g_, size_t f_);
  void fwd2d(size_t g_, size_t f_);
  void free();
};