#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

# global imports
import numpy as np

from dimarray import Dim
from timeseries import TimeSeries
from basewrapper import BaseWrapper

#import pdb

class Events(np.recarray):

    _required_fields = None
    _skip_field_check = False

    # def __new__(subtype, shape, dtype=None, buf=None, offset=0, strides=None,
    #             formats=None, names=None, titles=None,
    #             byteorder=None, aligned=False):
    def __new__(*args,**kwargs):
        self=np.recarray.__new__(*args,**kwargs)

        if not(self._required_fields is None):
            for req_field in self._required_fields:
                if not(req_field in self.dtype.names):
                    raise ValueError(
                        req_field+' is a required field!')
                if not(
                    self[req_field].dtype==self._required_fields[req_field]):
                    raise ValueError(
                        req_field+' needs to be '+
                        str(self._required_fields[req_field])+
                        '. Provided dtype was '+str(self[req_field].dtype))                             
        return self
    
    def remove_fields(self,*fields_to_remove):
        """
        Return a new instance of the array with specified fields
        removed.
        """
        if not(self._required_fields is None):
            for req_field in self._required_fields:
                if sum(map(lambda x: x==req_field,fields_to_remove)) > 0:
                    raise ValueError(req_field+' is a required field!')
        # sequence of arrays and names
        arrays = []
        names = ''

        # loop over fields, keeping if not matching fieldName
        for field in self.dtype.names:
            # don't add the field if in fields_to_remove list
            if sum(map(lambda x: x==field,fields_to_remove)) == 0:
                # append the data
                arrays.append(self[field])
                if len(names)>0:
                    # append ,
                    names = names+','
                # append the name
                names = names+field

        # return the new Events
        return np.rec.fromarrays(arrays,names=names).view(self.__class__)

    def add_fields(self,**fields):
        """
        Add fields from the keyword args provided and return a new
        instance.  To add an empty field, pass a dtype as the array.

        addFields(name1=array1, name2=dtype('i4'))
        
        """

        # list of current dtypes to which new dtypes will be added:
        # new_dtype = [(name,self[name].dtype) for name in self.dtype.names]
        
        # sequence of arrays and names from starting recarray
        arrays = map(lambda x: self[x], self.dtype.names)
        names = ','.join(self.dtype.names)
        
        # loop over the kwargs of field
        for name,data in fields.iteritems():
            # see if already there, error if so
            if self.dtype.fields.has_key(name):
                # already exists
                raise ValueError('Field "'+name+'" already exists.')
            
            # append the array and name
            if(isinstance(data,np.dtype)|
               isinstance(data,type)|isinstance(data,str)):
                # add empty array the length of the data
                arrays.append(np.empty(len(self),data))
            else:
                # add the data as an array
                arrays.append(data)

            # add the name
            if len(names)>0:
                # append ,
                names = names+','
            # append the name
            names = names+name
        # return the new Events
        return np.rec.fromarrays(arrays,names=names).view(self.__class__)

    def __getitem__(self, index):
        # try block to ensure the _skip_dim_check flag gets reset
        # in the following finally block
        try: 
            # skip the dim check b/c we're going to fiddle with them
            self._skip_field_check = True
            ret = np.recarray.__getitem__(self,index)
        finally:
            # reset the _skip_dim_check flag:
            self._skip_field_check = False
        return ret
            
    def __getslice__(self,i,j):
        try: 
            # skip the field check
            self._skip_field_check = True
            ret = np.recarray.__getslice__(self,i,j)           
        finally:
            # reset the _skip_field_check flag:
            self._skip_field_check = False
        return ret


    def __array_finalize__(self,obj):
        # print self.shape,obj.shape
        # # print self._skip_field_check,obj._skip_field_check
        self._skip_field_check = False
        # if this method is called with _skip_field_check == True, don't
        # check fields:
        # PBS: Also don't check if obj is None (implies it's unpickling)
        # We must find out if there are other instances of it being None
        if (isinstance(obj,Events) and obj._skip_field_check) or \
           obj is None: return
        # ensure that the fields are valid:
        if not(self._required_fields is None):
            for req_field in self._required_fields:
                if not(req_field in obj.dtype.names):
                    raise ValueError(
                        req_field+' is a required field!')
                if not(
                    obj[req_field].dtype==self._required_fields[req_field]):
                    raise ValueError(
                        req_field+' needs to be '+
                        str(self._required_fields[req_field])+
                        '. Provided dtype was '+str(obj[req_field].dtype))

class TsEvents(Events):
    """
    Class to hold time series events.  The record fields must include
    both tssrc (time series source) and tsoffset (time series offset)
    so that the class can know how to retrieve data for each event.
    """

    _required_fields = {'tssrc':BaseWrapper,'tsoffset':int}
        
    # def get_data(self,channel,dur,offset,buf,resampled_rate=None,
    #              filt_freq=None,filt_type='stop',
    #              filt_order=4,keep_buffer=False):
    def get_data(self,*args,**kwargs):
        """
        Return the requested range of data for each event by using the
        proper data retrieval mechanism for each event.

        The result will be an TimeSeries instance with dimensions
        (events,time) for the data and also some information about the
        data returned.
        """
	# get ready to load dat
	eventdata = None
        
        # events = self.data

        # speed up by getting unique event sources first
        usources = np.unique1d(self['tssrc'])

        # loop over unique sources
        for src in usources:
            # get the eventOffsets from that source
            ind = np.atleast_1d(self['tssrc']==src)
            
            if len(ind) == 1:
                src_events=self
            else:
                src_events = self[ind]

            #print "Loading %d events from %s" % (ind.sum(),src)
            # get the timeseries for those events            
            # newdat = src.get_event_data(channel,
            #                             src_events,
            #                             dur,
            #                             offset,
            #                             buf,
            #                             resampled_rate,
            #                             filt_freq,
            #                             filt_type,
            #                             filt_order,
            #                             keep_buffer)
            newdat = src.get_event_data(*args,**kwargs)

            # see if concatenate
            if eventdata is None:
                # start the new eventdata
                eventdata = newdat
            else:
                # append it to the existing
                np.concatenate(eventdata,newdat,eventdata.tdim)

        if eventdata is None:
            dims = [Dim(np.array(None), 'event'),
                    Dim(np.array(None), 'time')]
            eventdata = TimeSeries(np.atleast_2d(np.array(None)),
                                   samplerate=None,tdim='time',dims=dims)
        return eventdata
    

