
# global imports
import numpy as N

import string
import re
import sys

#import pdb

class Events(object):
    def __init__(self,data=None,dtype=None,**fields):

        if data is None:
            if dtype is None:
                # create from dtype
                raise NotImplementedError
            else:
                # create from fields
                raise NotImplementedError
        else:
            # passed a recarray or ndarray
            # XXX should this do a copy?
            self.data = data
        
    def __getitem__(self,item):
        return self.data[item]

    def __setitem__(self,item,value):
        self.data[item] = value

    def select(self,item):
        """
        Return a new instance of the class with specified slice of the data.
        """
        return self.__class__(self.data[item])
    
    def copy(self):
        """
        Return a copy of this Events instance.
        """
        if self.data is None:
            new_dat = None
        else:
            new_dat = self.data.copy()        
        return Events(new_dat)
    
    def extend(self,newrows):
        raise NotImplementedError
    
    def remove_fields(self,*fieldsToRemove):
        """
        Return a new instance of the array with specified fields
        removed.
        """
        # sequence of arrays and names
        arrays = []
        names = ''

        # loop over fields, keeping if not matching fieldName
        for field in self.data.dtype.names:
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

    def add_fields(self,**fields):
        """ Add fields from the keyword args provided and return a new
        instance.  To add an empty field, pass a dtype as the array.

        addFields(name1=array1, name2=dtype('i4'))
        
        """
        # sequence of arrays and names from starting recarray
        arrays = map(lambda x: self[x], self.data.dtype.names)
        names = string.join(self.data.dtype.names,',')
        
        # loop over the kwargs of field
        for name,data in fields.iteritems():
            # see if already there, error if so
            if self.data.dtype.fields.has_key(name):
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


class EegEvents(Events):
    """Class to hold EEG events.  The record fields must include both
eegsrc and eegoffset so that the class can know how to retrieve data
for each event."""
    
    def copy(self):
        """
        Return a copy of this EegEvents instance.
        """
        if self.data is None:
            new_dat = None
        else:
            new_dat = self.data.copy()        
        return EegEvents(new_dat)

    def get_data(self,channel,DurationMS,OffsetMS,BufferMS,resampledRate=None,
                  filtFreq=None,filtType='stop',filtOrder=4,keepBuffer=False):
        """
        Return the requested range of data for each event by using the
        proper data retrieval mechanism for each event.

        The result will be an EegTimeSeries instance with
        dimensions (events,time) for the data and also some
        information about the data returned.  """
	# get ready to load dat
	eventdata = None
        
        #if len(self.data.shape)==0:
	#    events = [self.data]
	#else:
        events = self.data

        # speed up by getting unique event sources first
        usources = N.unique1d(events['eegsrc'])

        # loop over unique sources
        for src in usources:
            # get the eventOffsets from that source
            ind = N.atleast_1d(events['eegsrc']==src)
            if len(ind) <= 1:
                if ind:
                    srcEvents=self.copy()
                else:
                    raise ValueError
            else:
                srcEvents = self.select(ind)

            #print "Loading %d events from %s" % (ind.sum(),src)
            # get the timeseries for those events            
            newdat = src.get_event_data(channel,
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
    

