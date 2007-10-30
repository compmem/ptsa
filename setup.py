

from distutils.core import setup, Extension
from distutils.sysconfig import get_config_var
import os
import sys

# get the version loaded as vstr
execfile('pyeeg/versionString.py')

# # set up the data files
# site_packages_dir = os.path.join(get_config_var('BINLIBDEST'), 'site-packages')
# data_files = []

# # The version string text file
# data_files.append((os.path.join(site_packages_dir, 'pyeeg'), 
# 		   ['src/versionString.txt']))


setup(name='pyeeg', 
      version=vstr, 
      package_dir={"pyeeg":"pyeeg"},
      packages=['pyeeg','pyeeg.tests'],
      author=['Per B. Sederberg, Christoph T. Weidemann'],
      maintainer=['Per B. Sederberg'],
      maintainer_email=['psederberg@gmail.com'],
      url=[''])

