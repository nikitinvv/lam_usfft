================
Installation
================
::

  conda create -n lam_usfft -c conda-forge cupy scikit-build swig tifffile h5py 
  git clone https://github.com/nikitinvv/lam_usfft
  cd lam_usfft
  pip install .


=============
Adjoint test:
=============
::

  cd tests;
  python test_chip.py
  (38077720000-1336.0157j) (38077747000-0.17443448j)
  195134.03
  
======================================
Performance test with nvidia profiling
======================================
::

  cd tests;
  nsys profile python test_perf.py
  # open the generated file with nsys-ui


  
  


