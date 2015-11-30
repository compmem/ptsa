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
from timeseries import TimeSeries,Dim

#import pdb

class Events(np.recarray):
    """
    A recarray with the events to be analyzed. Includes convenience
    functions to add and remove fields and a function to get a
    TimeSeries instance with the data linked to each event.
    """

    # def __new__(subtype, shape, dtype=None, buf=None, offset=0, strides=None,
    #             formats=None, names=None, titles=None,
    #             byteorder=None, aligned=False):
    
    # def __new__(*args,**kwargs):
    #     return np.recarray.__new__(*args,**kwargs)

    def __new__(subtype, data):
        return data.view(subtype)
        
    def remove_fields(self,*fields_to_remove):
        """
        Return a new instance of the recarray with specified fields
        removed.

        Parameters
        ----------
        *fields_to_remove : {list of strings}

        Returns
        -------
        New Events instance without the specified fields.
        """
        # sequence of arrays and names
        arrays = []
        names = []

        # loop over fields, keeping if not matching fieldName
        for field in self.dtype.names:
            # don't add the field if in fields_to_remove list
            #if sum(map(lambda x: x==field,fields_to_remove)) == 0:
            if not field in fields_to_remove:
                # append the data
                arrays.append(self[field])
                names.append(field)

        # return the new Events
        if len(arrays) == 0:
            arrays.append([])
        return np.rec.fromarrays(arrays,names=','.join(names)).view(self.__class__)
        
    def add_fields(self,**fields):
        """
        Add fields from the keyword args provided and return a new
        instance.

        Parameters
        ----------
        **fields_to_add : {dictionary}
            Names in the dictionary correspond to new field names and
            the values specify their content. To add an empty field,
            pass a dtype as the value.

        Returns
        -------
        New Events instance with the specified new fields.
        
        Examples
        --------
        events.add_fields(name1=array1, name2=dtype('i4'))
        
        """

        # list of current dtypes to which new dtypes will be added:
        # new_dtype = [(name,self[name].dtype) for name in self.dtype.names]
        
        # sequence of arrays and names from starting recarray
        #arrays = map(lambda x: self[x], self.dtype.names)
        arrays = [self[x] for x in self.dtype.names]
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

    def get_data(self,channels,start_time,end_time,buffer_time=0.0,
                 resampled_rate=None,
                 filt_freq=None,filt_type='stop',filt_order=4,
                 keep_buffer=False,esrc='esrc',eoffset='eoffset',
                 loop_axis=None,num_mp_procs=0,
                 eoffset_in_time=True):
        """
        Return the requested range of data for each event by using the
        proper data retrieval mechanism for each event.

        Parameters
        ----------
        channels: {list,int,None}
            Channels from which to load data.
        start_time: {float}
            Start of epoch to retrieve (in time-unit of the data).
        end_time: {float}
            End of epoch to retrieve (in time-unit of the data).
        buffer_time: {float},optional
            Extra buffer to add on either side of the event in order
            to avoid edge effects when filtering (in time unit of the
            data).
        resampled_rate: {float},optional
            New samplerate to resample the data to after loading.
        filt_freq: {array_like},optional
            The range of frequencies to filter (depends on the filter
            type.)
        filt_type = {scipy.signal.band_dict.keys()},optional
            Filter type.
        filt_order = {int},optional
            The order of the filter.
        keep_buffer: {boolean},optional
            Whether to keep the buffer when returning the data.
        esrc : {string},optional
            Name for the field containing the source for the time
            series data corresponding to the event.
        eoffset: {string},optional
            Name for the field containing the offset (in seconds) for
            the event within the specified source.
        eoffset_in_time: {boolean},optional        
            If True, the unit of the event offsets is taken to be
            time (unit of the data), otherwise samples.
        
        Returns
        -------
        A TimeSeries instance with dimensions (channels,events,time).
        """
        
        # check for necessary fields
        if not (esrc in self.dtype.names and
                eoffset in self.dtype.names):
            raise ValueError(esrc+' and '+eoffset+' must be valid fieldnames '+
                             'specifying source and offset for the data.')
        
	# get ready to load dat
	eventdata = []
        events = []
        
        # speed up by getting unique event sources first
        usources = np.unique(self[esrc])

        # loop over unique sources
        eventdata = None
        for src in usources:
            # get the eventOffsets from that source
            ind = np.atleast_1d(self[esrc]==src)
            
            if len(ind) == 1:
                event_offsets=self[eoffset]
                events.append(self)
            else:
                event_offsets = self[ind][eoffset]
                events.append(self[ind])

            #print "Loading %d events from %s" % (ind.sum(),src)
            # get the timeseries for those events
            newdat = src.get_event_data(channels,
                                        event_offsets,
                                        start_time,
                                        end_time,
                                        buffer_time,
                                        resampled_rate,
                                        filt_freq,
                                        filt_type,
                                        filt_order,
                                        keep_buffer,
                                        loop_axis,
                                        num_mp_procs,
                                        eoffset,
                                        eoffset_in_time)
            if eventdata is None:
                eventdata = newdat
            else:
                eventdata = eventdata.extend(newdat,axis=1)
            
        # concatenate (must eventually check that dims match)
        tdim = eventdata['time']
        cdim = eventdata['channels']
        srate = eventdata.samplerate
        events = np.concatenate(events).view(self.__class__)
        eventdata = TimeSeries(eventdata,
                               'time', srate,
                               dims=[cdim,Dim(events,'events'),tdim])
        
        return eventdata

    

