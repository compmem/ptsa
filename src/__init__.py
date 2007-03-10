"""
PyEEG - The Python EEG toolbox.
"""


from data import DataArray,DataDict,DataWrapper,Events,RawBinaryEEG,createEventsFromMatFile
from filter import buttfilt, decimate, filtfilt
from plotting import topoplot
from wavelet import tfPhasePow,phasePow1d,phasePow2d
from version import versionAtLeast,versionWithin

#__all__ = [data,filter,plotting,wavelet]

