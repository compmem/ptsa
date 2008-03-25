#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

"""
Version management module.
"""

from distutils.version import StrictVersion

## !!!!!
import versionString
#vstr = open('versionString.txt').readline().strip()
## !!!!!

ptsaVersion = StrictVersion(versionString.vstr)

def versionAtLeast(someString):
    """
    Check that the current ptsa Version >= argument string's version.
    """
    if ptsaVersion >= StrictVersion(someString):
        # Is above specified version
        return True
    else:
        return False

def versionWithin(str1, str2):
    """
    Check that the current ptsa version is in the version-range described
    by the 2 argument strings.
    """
    if not (ptsaVersion >= StrictVersion(str1) and ptsaVersion <= StrictVersion(str2)):
        # not within range
        return False
    else:
        # within range
        return True
