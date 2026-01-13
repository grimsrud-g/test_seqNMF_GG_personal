Lmod Warning:
-------------------------------------------------------------------------------
The following dependent module(s) are not currently loaded: python/2.7.13
(required by: py-numpy/1.14.3_py27)
-------------------------------------------------------------------------------




The following have been reloaded with a version change:
  1) python/2.7.13 => python/3.12.1     2) sqlite/3.18.0 => sqlite/3.44.2

Lmod Warning:
-------------------------------------------------------------------------------
The following dependent module(s) are not currently loaded: python/2.7.13
(required by: py-numpy/1.14.3_py27)
-------------------------------------------------------------------------------



Traceback (most recent call last):
  File "/share/software/user/open/py-numpy/1.14.3_py27/lib/python2.7/site-packages/numpy/core/__init__.py", line 16, in <module>
    from . import multiarray
ImportError: /share/software/user/open/py-numpy/1.14.3_py27/lib/python2.7/site-packages/numpy/core/multiarray.so: undefined symbol: _Py_ZeroStruct

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/oak/stanford/groups/anishm/grimsrud/test_seqNMF/code/refit_fmri_jNMF_Babies_rest.py", line 7, in <module>
    import numpy as np
  File "/share/software/user/open/py-numpy/1.14.3_py27/lib/python2.7/site-packages/numpy/__init__.py", line 142, in <module>
    from . import add_newdocs
  File "/share/software/user/open/py-numpy/1.14.3_py27/lib/python2.7/site-packages/numpy/add_newdocs.py", line 13, in <module>
    from numpy.lib import add_newdoc
  File "/share/software/user/open/py-numpy/1.14.3_py27/lib/python2.7/site-packages/numpy/lib/__init__.py", line 8, in <module>
    from .type_check import *
  File "/share/software/user/open/py-numpy/1.14.3_py27/lib/python2.7/site-packages/numpy/lib/type_check.py", line 11, in <module>
    import numpy.core.numeric as _nx
  File "/share/software/user/open/py-numpy/1.14.3_py27/lib/python2.7/site-packages/numpy/core/__init__.py", line 26, in <module>
    raise ImportError(msg)
ImportError: 
Importing the multiarray numpy extension module failed.  Most
likely you are trying to import a failed build of numpy.
If you're working with a numpy git repo, try `git clean -xdf` (removes all
files not under version control).  Otherwise reinstall numpy.

Original error was: /share/software/user/open/py-numpy/1.14.3_py27/lib/python2.7/site-packages/numpy/core/multiarray.so: undefined symbol: _Py_ZeroStruct

