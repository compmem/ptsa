from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

sourcefiles = ["edf.pyx", "edfwrap.c", "edflib.c"]
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("edf",
                             sources = sourcefiles,
                             define_macros = [('_LARGEFILE64_SOURCE', None),
                                              ('_LARGEFILE_SOURCE', None)]),
                   ]
)
