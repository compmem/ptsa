#emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


# methods for aligning events and eeg data

import os
import csv
import numpy as np

from edfwrapper import EdfWrapper

def load_pyepl_eeg_pulses(logfile):
    # open and read each line
    reader = csv.reader(open(logfile,'rU'),dialect=csv.excel_tab)
    pulses = []
    for row in reader:
        if row[2] == 'UP':
            pulses.append(long(row[0]))
    return np.asarray(pulses)

def find_needle_in_haystack(needle, haystack, maxdiff):
    """
    """
    nlen = len(needle)
    found = False
    for i in range(len(haystack)-nlen):
        if (haystack[i:i+nlen] - needle).max() < maxdiff:
            found = True
            break
    if not found:
        i = None
    return i

def align_edf_pyepl(edffile, eeglog, events, annot_id='S255', 
                    channel_for_sr=0, 
                    window=100, thresh_ms=10,
                    event_time_id='event_time'):
    """
    """
    # create the edf wrapper
    ew = EdfWrapper(edffile)

    # load clean pyepl eeg log
    pulse_ms = load_pyepl_eeg_pulses(eeglog)

    # load annotations from edf
    annot = ew.get_annotations()

    # convert seconds to ms for annot_ms
    annot_ms = annot[annot['annotations']==annot_id]['onsets'] * 1000

    # pick beginning and end (needle in haystack)
    for i in xrange(len(annot_ms)-window):
        s_ind = find_needle_in_haystack(np.diff(annot_ms[i:i+window]),
                                        np.diff(pulse_ms),thresh_ms)
        if not s_ind is None:
            break
    if s_ind is None:
        raise ValueError("Unable to find a start window.") # get better error here
    start_annot_vals = annot_ms[i:i+window]
    start_pulse_vals = pulse_ms[s_ind:s_ind+window]

    for i in xrange(len(annot_ms)-window):
        e_ind = find_needle_in_haystack(np.diff(annot_ms[::-1][i:i+window]),
                                        np.diff(pulse_ms[::-1]),thresh_ms)
        if not e_ind is None:
            break
    if e_ind is None:
        raise ValueError("Unable to find a end window.") # get better error here

    # correct the end ind 
    e_ind = len(pulse_ms) - e_ind - window

    i = len(annot_ms) - i - window
    end_annot_vals = annot_ms[i:i+window]
    end_pulse_vals = pulse_ms[e_ind:e_ind+window]

    # fit line with regression
    x = np.r_[start_pulse_vals,end_pulse_vals]
    y = np.r_[start_annot_vals,end_annot_vals]
    m,c = np.linalg.lstsq(np.vstack([x-x[0],np.ones(len(x))]).T, y)[0]
    c = c - x[0]*m

    # calc the event time in offsets
    samplerate = ew.get_samplerate(channel)
    offsets = long(np.round((m*events[event_time_id] + c)*samplerate/1000.))
    
    # add esrc and eoffset to the Events instance
    events = events.add_fields(esrc=np.repeat(ew,len(events)),
                               eoffset=offsets)

    # return the updated events
    return events

