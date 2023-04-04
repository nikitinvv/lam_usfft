/*interface*/
%module copy3d

%{
#define SWIG_FILE_WITH_INIT
#include "copy3d.cuh"
%}

class copy3d {

public:  
  %mutable;  
  copy3d();
  ~copy3d();  
  void copyh2d(size_t f_);
  void free();
};