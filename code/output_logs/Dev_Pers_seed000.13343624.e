Lmod Warning:
-------------------------------------------------------------------------------
The following dependent module(s) are not currently loaded: python/3.14.2
(required by: py-numpy/2.3.5_py314)
-------------------------------------------------------------------------------




The following have been reloaded with a version change:
  1) gcc/14.2.0 => gcc/12.4.0           4) sqlite/3.51.1 => sqlite/3.44.2
  2) libffi/3.4.5 => libffi/3.2.1       5) zlib/1.3.1 => zlib/1.2.11
  3) python/3.14.2 => python/3.12.1

Lmod Warning:
-------------------------------------------------------------------------------
The following dependent module(s) are not currently loaded: python/3.14.2
(required by: py-numpy/2.3.5_py314)
-------------------------------------------------------------------------------



Traceback (most recent call last):
  File "/share/software/user/open/py-numpy/2.3.5_py314/lib/python3.14/site-packages/numpy/_core/__init__.py", line 22, in <module>
    from . import multiarray
  File "/share/software/user/open/py-numpy/2.3.5_py314/lib/python3.14/site-packages/numpy/_core/multiarray.py", line 11, in <module>
    from . import _multiarray_umath, overrides
  File "/share/software/user/open/py-numpy/2.3.5_py314/lib/python3.14/site-packages/numpy/_core/overrides.py", line 5, in <module>
    from numpy._core._multiarray_umath import (
ModuleNotFoundError: No module named 'numpy._core._multiarray_umath'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/share/software/user/open/py-numpy/2.3.5_py314/lib/python3.14/site-packages/numpy/__init__.py", line 112, in <module>
    from numpy.__config__ import show_config
  File "/share/software/user/open/py-numpy/2.3.5_py314/lib/python3.14/site-packages/numpy/__config__.py", line 4, in <module>
    from numpy._core._multiarray_umath import (
  File "/share/software/user/open/py-numpy/2.3.5_py314/lib/python3.14/site-packages/numpy/_core/__init__.py", line 48, in <module>
    raise ImportError(msg) from exc
ImportError: 

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the numpy C-extensions failed. This error can happen for
many reasons, often due to issues with your setup or how NumPy was
installed.

We have compiled some common reasons and troubleshooting tips at:

    https://numpy.org/devdocs/user/troubleshooting-importerror.html

Please note and check the following:

  * The Python version is: Python3.12 from "/share/software/user/open/python/3.12.1/bin/python3"
  * The NumPy version is: "2.3.5"

and make sure that they are the versions you expect.
Please carefully study the documentation linked above for further help.

Original error was: No module named 'numpy._core._multiarray_umath'


The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/oak/stanford/groups/anishm/grimsrud/test_seqNMF/code/refit_fmri_jNMF_Babies_rest.py", line 7, in <module>
    import numpy as np
  File "/share/software/user/open/py-numpy/2.3.5_py314/lib/python3.14/site-packages/numpy/__init__.py", line 117, in <module>
    raise ImportError(msg) from e
ImportError: Error importing numpy: you should not try to import numpy from
        its source directory; please exit the numpy source tree, and relaunch
        your python interpreter from there.
