"""
PyEEG - The Python EEG toolbox.
"""


#from data import DataWrapper,Events,RawBinaryEEG,createEventsFromMatFile
from data import Dim,Dims,DimData,EegTimeSeries
from filter import buttfilt, filtfilt
from plotting import topoplot
from wavelet import tsPhasePow,phasePow1d,phasePow2d
from version import versionAtLeast,versionWithin

#__all__ = [data,filter,plotting,wavelet]

packages = ('pyeeg',
            'pyeeg.tests')

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
    importall('pyeeg')
    return getattr(NumpyTest(), method)(level, verbosity=2)

def test(level=1, verbosity=1, flags=[]):
    _test('test', level=level, verbosity=verbosity, flags=flags)
test.__doc__ = "Using NumpyTest test method.\n"+_test.__doc__

def testall(level=1, verbosity=1, flags=[]):
    _test('testall', level=level, verbosity=verbosity, flags=flags)
testall.__doc__ = "Using NumpyTest testall method.\n"+_test.__doc__

