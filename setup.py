

from distutils.core import setup, Extension
from distutils.sysconfig import get_config_var
import os
import sys

# get the version loaded as vstr
execfile('ptsa/versionString.py')

setup(name='ptsa', 
      version=vstr, 
      packages=['ptsa','ptsa.tests','ptsa.data','ptsa.data.tests',
                'ptsa.plotting','ptsa.plotting.tests',
                'dimarray','dimarray.tests'],
      maintainer=['Per B. Sederberg'],
      maintainer_email=['psederberg@gmail.com'],
      url=['http://ptsa.sourceforge.net'])

