

from distutils.core import setup, Extension
from distutils.sysconfig import get_config_var
import os
import sys

# get the version loaded as vstr
execfile('ptsa/versionString.py')

# # set up the data files
# site_packages_dir = os.path.join(get_config_var('BINLIBDEST'), 'site-packages')
# data_files = []

# # The version string text file
# data_files.append((os.path.join(site_packages_dir, 'ptsa'), 
# 		   ['src/versionString.txt']))


setup(name='ptsa', 
      version=vstr, 
      package_dir={"ptsa":"ptsa"},
      packages=['ptsa','ptsa.tests','ptsa.data','ptsa.data.tests'],
      author=['Per B. Sederberg, Christoph T. Weidemann'],
      maintainer=['Per B. Sederberg'],
      maintainer_email=['psederberg@gmail.com'],
      url=[''])

