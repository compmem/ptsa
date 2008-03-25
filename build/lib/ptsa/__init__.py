#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

"""
PTSA - The Python Time-Series Analysis toolbox.
"""


#from data import DataWrapper,Events,RawBinaryEEG,createEventsFromMatFile
from data import Dim,Dims,DimData,TimeSeries
from filt import buttfilt, filtfilt
from plotting import topoplot
from wavelet import tsPhasePow,phasePow1d,phasePow2d
from version import versionAtLeast,versionWithin
from fixed_scipy import wavelets

#__all__ = [data,filter,plotting,wavelet]

packages = ('ptsa',
            'ptsa.tests',
            'ptsa.data',
            'ptsa.data.tests',
            'ptsa.fixed_scipy')

def _test(method, level, verbosity, flags):
    """
    Run test suite with level and verbosity.

        level:
          None           --- do nothing, return None
          < 0            --- scan for tests of level=abs(level),
                             don't run them, return TestSuite-list
          > 0            --- scan for tests of level, run them,
                             return TestRunner

        verbosity:
          >= 0           --- show information messages
          > 1            --- show warnings on missing tests
    """
    from numpy.testing import NumpyTest, importall
    #from neuroimaging.utils.testutils import set_flags
    #set_flags(flags)
    importall('ptsa')
    return getattr(NumpyTest(), method)(level, verbosity=2)

def test(level=1, verbosity=1, flags=[]):
    _test('test', level=level, verbosity=verbosity, flags=flags)
test.__doc__ = "Using NumpyTest test method.\n"+_test.__doc__

def testall(level=1, verbosity=1, flags=[]):
    _test('testall', level=level, verbosity=verbosity, flags=flags)
testall.__doc__ = "Using NumpyTest testall method.\n"+_test.__doc__

