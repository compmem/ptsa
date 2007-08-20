# PyEEG: version.py
#
# Copyright (C) 2007 Per B. Sederberg
# Authors: Per B. Sederberg, Christoph T. Weidemann
# URL: 
#
# Distributed under the terms of the GNU Lesser General Public License
# (LGPL). See the license.txt that came with this file.

"""
Version management module.
"""

from distutils.version import StrictVersion

## !!!!!
import versionString
#vstr = open('versionString.txt').readline().strip()
## !!!!!

pyeegVersion = StrictVersion(versionString.vstr)

def versionAtLeast(someString):
    """
    Check that the current pyeeg Version >= argument string's version.
    """
    if pyeegVersion >= StrictVersion(someString):
        # Is above specified version
        return True
    else:
        return False

def versionWithin(str1, str2):
    """
    Check that the current pyeeg version is in the version-range described
    by the 2 argument strings.
    """
    if not (pyeegVersion >= StrictVersion(str1) and pyeegVersion <= StrictVersion(str2)):
        # not within range
        return False
    else:
        # within range
        return True
