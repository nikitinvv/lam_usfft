# lam_usfft



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


======================================
Performance test with nvidia profiling
======================================
::

  cd tests;
  python test_chip.py
  nsys profile python test_perf.py
  # open the generated file with nsys-ui


  
  


