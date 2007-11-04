
# global imports
import numpy as N

from scipy.io import loadmat
from scipy.signal import resample

import string
import re
import sys

#import pdb

# Define exceptions:
class DataException(Exception): pass
class EventsMatFileError(DataException): pass
class FilterStringError(DataException): pass

# data array subclass of recarray
class DataArray(N.recarray):
    """ Class that extends the record array so that it's easy to filter
    your records based on various conditions.  """
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
                # prepend "self." to fields in event structure
                # RE makes sure to not replace substrings
                filterStr = re.sub(r'\b'+k+r'\b','self.'+k,filterStr)
                
            # eval to set the boolean indices
            try:
                ind = eval(filterStr)
            except:
                raise FilterStringError, str(sys.exc_info()[0])+"\n"+str(sys.exc_info()[1])+"\nThe filter string is not a valid Python expression!\nExample string:\n\'(var1==0) & (var2<=0)\'"
        else:
            # must iterate over each one to get indices
            for k in self.dtype.names:
                # prepend "self." and append "[i]" to fields in event structure
                # RE makes sure to not replace substrings
                filterStr = re.sub(r'\b'+k+r'\b','self.'+k+'[i]',filterStr)
                
            # apply filter to each item
            try:
                ind = eval(filterStr)
            except:
                raise FilterStringError, str(sys.exc_info()[0])+"\n"+str(sys.exc_info()[1])+"\nThe filter string is not a valid Python expression!\nExample string:\n\'(var1==0) & (var2<=0)\'"

        # return the ind as an array
        return N.asarray(ind)

    def filter(self,filterStr,iterate=False):
        """
        Run filterInd and return its application to the records. 

	Note that this does not make a copy of the data, it only
	slices it."""
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
        """ Add fields from the keyword args provided and return a new
        instance.  To add an empty field, pass a dtype as the array.

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

	
class Events(DataArray):
    """Class to hold EEG events.  The record fields must include both
eegsrc and eegoffset so that the class can know how to retrieve data
for each event."""
    def getDataMS(self,channel,DurationMS,OffsetMS,BufferMS,resampledRate=None,filtFreq=None,filtType='stop',filtOrder=4,keepBuffer=False):
        """
        Return the requested range of data for each event by using the
        proper data retrieval mechanism for each event.

        The result will be a dictionary with an EEG array of
        dimensions (events,time) for the data and also some
        information about the data returned.  """
	# get ready to load dat
	eventdata = None
        
        if len(self.shape)==0:
	    events = [self]
	else:
	    events = self

        # speed up by getting unique event sources first
        usources = N.unique1d(events['eegsrc'])

        # loop over unique sources
        for src in usources:
            # get the eventOffsets from that source
            ind = events['eegsrc']==src
            evOffsets = events['eegoffset'][ind]
            srcEvents = events[ind]
                                      
            # get the timeseries for those events
            newdat = src.getDataMS(channel,
                                   srcEvents,
                                   DurationMS,
                                   OffsetMS,
                                   BufferMS,
                                   resampledRate,
                                   filtFreq,
                                   filtType,
                                   filtOrder,
                                   keepBuffer)

            # see if concatenate
            if eventdata is None:
                # start the new eventdata
                eventdata = newdat
            else:
                # append it to the existing
                eventdata.extend(newdat,0)

        return eventdata

# 	# loop over events
# 	for evNo,ev in enumerate(events):
# 	    # get the eeg
# 	    newdat = ev['eegsrc'].getDataMS(channel,
# 					    ev['eegoffset'],
# 					    DurationMS,
# 					    OffsetMS,
# 					    BufferMS,
# 					    resampledRate,
# 					    filtFreq,
# 					    filtType,
# 					    filtOrder,
# 					    keepBuffer)

# 	    # allocate if necessary
# 	    if len(eventdata) == 0:
# 		# make ndarray with events by time
# 		eventdata = N.empty((len(events),newdat['data'].shape[1]),
# 				    dtype=newdat['data'].dtype)

# 	    # fill up the eventdata
# 	    eventdata[evNo,:] = newdat['data'][0,:]

# 	# set the newdat to hold all the data
# 	newdat['data'] = eventdata

# 	# force uniform samplerate, so if no resampledRate is
# 	# provided, fix to samplerate of first event.

# 	# return (events, time) ndarray with samplerate
# 	return newdat

#     # we need to make sure and return an our custom class if it's a
#     # single record, this will ensure we can still call our custom
#     # methods
#     def __getitem__(self, indx):
#         obj = N.ndarray.__getitem__(self, indx)
#         if (isinstance(obj, N.ndarray) and obj.dtype.isbuiltin):
#             return obj.view(N.ndarray)
# 	elif isinstance(obj, N.record):
# 	    # return record as our custom recarray XXX Eventually this
# 	    # should return a EventRecord instance.
# 	    return self.__class__(obj)
#         return obj

	
def createEventsFromMatFile(matfile):
    """Create an events data array from an events structure saved in a
    Matlab mat file."""
    # load the mat file
    mat = loadmat(matfile)

    if 'events' not in mat.keys():
        raise EventsMatFileError, "\nError processing the Matlab file: %s\nThis file must contain an events structure with the name \"events\" (case sensitive)!\n(All other content of the file is ignored.)" % matfile 
    
    # get num events
    numEvents = len(mat['events'])

    # determine the fieldnames and formats
    fields = mat['events'][0]._fieldnames
    
    # create list with array for each field
    data = []
    hasEEGInfo = False
    for f,field in enumerate(fields):
	# handle special cases
	if field == 'eegfile':
	    # we have eeg info
	    hasEEGInfo = True

	    # get unique files
	    eegfiles = N.unique(map(lambda x: str(x.eegfile),mat['events']))
	    
	    # make dictionary of data wrapers for the eeg files
	    efile_dict = {}
	    for eegfile in eegfiles:
		efile_dict[eegfile] = RawBinaryEEG(eegfile)

	    # Handle when the eegfile field is blank
	    efile_dict[''] = None
	
	    # set the eegfile to the correct data wrapper
	    newdat = N.array(map(lambda x: efile_dict[str(x.__getattribute__(field))],
				 mat['events']))
			
	    # change field name to eegsrc
	    fields[f] = 'eegsrc'
	else:
	    # get the data in normal fashion
	    newdat = N.array(map(lambda x: x.__getattribute__(field),mat['events']))

	# append the data
	data.append(newdat)

    # allocate for new array
    newrec = N.rec.fromarrays(data,names=fields)

    # see if process into DataArray or Events
    if hasEEGInfo:
	#newrec = Events(newrec)
	newrec = newrec.view(Events)
    else:
	#newrec = DataArray(newrec)
	newrec = newrec.view(DataArray)
    return newrec



