
# necessary global imports
import numpy as N
import os
import glob
import string

import pdb



class DataWrapper:
    """
    Base class to provide interface to timeseries data.
    """
    def getdataMS(channels,eventOffsets,DurationMS,OffsetMS,BufferMS,resampledRate=None):
        pass

class BinaryEEG(DataWrapper):
    """
    Interface to data stored in binary format with a separate file for
    each channel.
    """
    def __init__(dataroot,samplerate=None,format=None,gain=None):
        # set up the basic params of the data

        # see if can find them from a params file in dataroot
        pass

    def getdataMS(channels,eventOffsets,DurationMS,OffsetMS,BufferMS,resampledRate=None):
        pass

class EEGData(N.array):
    def __new__(subtype,obj,samplerate,channels=None):
        self = obj.view(subtype)
        self.samplerate = samplerate
        self.channels = channels
        return self

class Events(DataArray):
    def align(self,eegPulseFiles,behPulseFiles,eegDataWrappers,channels,msField='mstime'):
        """
        Align the behavioral events with eeg data.
        """
        # will fill the list of data wrappers
        self.dataWrappers = []

        # determine the eeg offsets into the file for each event
        
        # determine the index into dataWrappers for each event (-1 for none)

    def getDataMS(channels,DurationMS,OffsetMS,BufferMS,resampledRate):
        """
        Return the requested range of data for each event by using the
        proper data retrieval mechanism for each event.

        The result will be an EEG array of dimensions (channels,events,time).
        """
        pass
        

# data array subclass of recarray
class DataArray(N.recarray):
    def __new__(subtype,obj):
        self = obj.view(subtype)
        return self
    def filterIndex(self,filterStr,iterate=False):
        """
        Return the boolean index to filter events based on a filter
        string.  You only need to iterate if there is not a ufunc for
        your test.  For example, if you want to perform a string.find
        on a field in the events.
        """
        # see if must iterate or vectorize
        if not iterate:
            # vectorize, so replace fields
            for k in self.dtype.names:
                filterStr = filterStr.replace(k,'self.'+k)

            # eval to set the boolean indices
            ind = eval(filterStr)
        else:
            # must iterate over each one to get indices
            for k in self.dtype.names:
                filterStr = filterStr.replace(k,'self.'+k+'[i]')

            # apply filter to each item
            ind = [eval(filterStr) for i in xrange(self.len())]

        # return the ind as an array
        return N.asarray(ind)

    def filter(self,filterStr,iterate=False):
        """
        Run filterInd and return its application to the events.
        """
        # get the filter index
        ind = self.filterIndex(filterStr,iterate)

        # apply the filter
	return self[ind]

    def removeFields(self,*fieldsToRemove):
        """
        Return a new instance of the array with specified fields
        removed.
        """
        # sequence of arrays and names
        arrays = []
        names = ''

        # loop over fields, keeping if not matching fieldName
        for field in self.dtype.names:
            # don't add the field if in fieldsToRemove list
            if sum(map(lambda x: x==field,fieldsToRemove)) == 0:
                # append the data
                arrays.append(self[field])
                if len(names)>0:
                    # append ,
                    names = names+','
                # append the name
                names = names+field

        # return the new recarray
        return self.__class__(N.rec.fromarrays(arrays,names=names))

    def addFields(self,**fields):
        """
        Add fields from the keyword args provided.  To add an empty
        field, pass a dtype as the array.

        addFields(name1=array1, name2=dtype('i4'))
        
        """
        # sequence of arrays and names from starting recarray
        arrays = map(lambda x: self[x], self.dtype.names)
        names = string.join(self.dtype.names,',')
        
        # loop over the kwargs of field
        for name,data in fields.iteritems():
            # see if already there, error if so
            if self.dtype.fields.has_key(name):
                # already exists
                raise AttributeError, 'Field "'+name+'" already exists.'
            
            # append the array and name
            if isinstance(data,N.dtype):
                # add empty array the length of the data
                arrays.append(N.empty(len(self),data))
            else:
                # add the data as an array
                arrays.append(data)

            # add the name
            if len(names)>0:
                # append ,
                names = names+','
            # append the name
            names = names+name

        # return the new recarray
        return self.__class__(N.rec.fromarrays(arrays,names=names))

