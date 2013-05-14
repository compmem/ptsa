#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from ptsa.peak_finder import peak_finder

def add_eog_events(events, eye_channels, extrema, edge = 120.,
                   min_std = 6., thresh = 5., max_thresh = 100.,
                   min_thresh=60., freq_range=[1,10],
                   esrc = 'esrc', eoffset = 'eoffset'):
    """
    Function for adding EOG events to an event structure.
    
    Parameters
    ----------
    events : Event
        Events structure to which EOG events should be added.
    eye_channels : dict
        Dictionary with channel labels as keys and channel numbers as values.
        If more than one channel is given, the channel with the highest std.
        is chosen.
    extrema : {1, -1}
        Determines if maxima (1) or minima (-1) are detected.
    edge : float, optional
        Time (in units of the data) to remove from beginning and end in
        when determining thresholds etc. (to avoid noise at the edges).
    min_std : float, optional
        If the std. of the EOG channel is below this value an error is raised.
    thresh : float, optional
        Threshold for peak detection (see peak_finder for details).
    max_thresh : float, optional
        Threshold for peak detection is capped at this value.
    min_thresh : float, optional
        If threshold falls short of this value, an error is raised.
    freq_range : list, optional
        List with lower and upper frequencies for filtering the EOG channel(s).    
    esrc : string, optional
        Name for the field containing the source for the time
        series data corresponding to the event.
    eoffset: string, optional
        Name for the field containing the offset (in samples) for
        the event within the specified source.    

    Returns
    -------
    events : Events
        Event structure with the following additional fields:
        * pre_artifact: closest time before event with EOG artifact 
        * post_artifact: closest time after event with EOG artifact
        * eog_label: label of the EOG channel
        * eog_thresh: Threshold used for artifact detection
        * eog_extrema: Extrema (peaks [1], or troughs [-1]) used for
        artifact detection.

    Notes
    -----
    Inspired by related code in the MNE package
    http://www.martinos.org/mne/
    https://github.com/mne-tools/mne-python
    """
    
    datasources = np.unique(events[esrc])

    # Initialize arrays for new fields:
    all_prepost_artifact = np.ones((len(events),2))*np.nan
    eog_labels = np.zeros(len(events),np.array(eye_channels.keys()).dtype)
    eog_thresh = np.zeros(len(events))
    eog_extrema = np.zeros(len(events))
    
    for data in datasources:
        eog = {}
        eog_std = {}
        max_std = 0
        max_label = None
        for chan in eye_channels:
            # We want the filtered EOG channel for easier artifact detection
            eog[chan] = data.data[eye_channels[chan],:].filtered(
                freq_range = freq_range, filt_type = 'band', order = 1)
            # We pick the channel with the largest std.:
            eog_std[chan] = np.std(eog[chan][
                eog[chan].samplerate*edge:-eog[chan].samplerate*edge])
            if eog_std[chan] > max_std:
                max_std = eog_std[chan]
                max_label = chan
        
        # Sanity check that EOG channel is OK:
        if max_std < min_std:
            raise ValueError('Max std. too low: '+str(max_std))

        # calculate threshold for peak_finder
        thresh = np.min([
            (eog[max_label][eog[max_label].samplerate*edge:-eog[
                max_label].samplerate*edge].max()-
             eog[max_label][eog[max_label].samplerate*edge:-eog[
                 max_label].samplerate*edge].min())/
            thresh,max_thresh])

        # Sanity check that EOG channel is OK (if range is too low,
        # threshold will be too small):
        if thresh < min_thresh:
            raise ValueError('Thresh too low: '+str(thresh))

        # # Originally let allowed extrema to be None with automatic
        # # detection as below, but should probably be specified.
        # if extrema is None:
        #     temp = eog[max_label][eog[max_label].samplerate*edge:-eog[
        #         max_label].samplerate*edge] - np.mean(eog[max_label][
        #             eog[max_label].samplerate*edge:-eog[
        #                 max_label].samplerate*edge])
        #     if np.abs(np.max(temp)) > np.abs(np.min(temp)):
        #         extrema = 1
        #     else:
        #         extrema = -1


        # Find extrema in the EOG channel
        eog_events,eog_mags = peak_finder(eog[max_label], extrema = extrema,
                                          thresh = thresh)

        # To determine the time of the artifacts before and after each
        # event we add the beginning and end of the recording in case
        # there are no other artifacts before/after the first/last
        # event.
        artifact = np.r_[0,eog_events,len(eog[max_label])]
        offsets = events[events[esrc] == data][eoffset]

        # If the data recording started after experiment onset or
        # stopped too early, offsets may be negative or exceed
        # recording lenght:
        prepost_artifact = np.array([
            (np.max(artifact[artifact<offset])-np.float(offset),
             np.min(artifact[artifact>=offset])-np.float(offset))
            for offset in offsets[(offsets>0)&
                                  (offsets<len(eog[max_label]))]])
        # For those cases we set the time between the event and the
        # artifact to zero:
        for i in range(np.sum(offsets<=0)):
            prepost_artifact = np.r_[[(0.,0.)], prepost_artifact]
        for i in range(np.sum(offsets>=len(eog[max_label]))):
            prepost_artifact = np.r_[prepost_artifact,[(0.,0.)]]
        prepost_artifact /= data.samplerate

        # Populate the arrays:
        all_prepost_artifact[events[esrc]==data] = prepost_artifact
        eog_labels[events[esrc]==data] = max_label
        eog_thresh[events[esrc]==data] = thresh
        eog_extrema[events[esrc]==data] = extrema

    # Ammend the event structure and return:
    events = events.add_fields(pre_artifact = all_prepost_artifact[:,0])
    events = events.add_fields(post_artifact = all_prepost_artifact[:,1])
    events = events.add_fields(eog_label=eog_labels)
    events = events.add_fields(eog_thresh=eog_thresh)
    events = events.add_fields(eog_extrema=eog_extrema)
    return(events)

        