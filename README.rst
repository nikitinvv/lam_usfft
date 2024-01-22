================
Installation
================
::

  conda create -n lam_usfft -c conda-forge cupy scikit-build swig  
  git clone https://github.com/nikitinvv/lam_usfft
  cd lam_usfft
  pip install .


=============
Adjoint test:
=============
::

  cd tests;
  python test_adjoint.py
  
