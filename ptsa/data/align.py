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
from basewrapper import BaseWrapper


def times_to_offsets(eeg_times, beh_times, ev_times, blen=10, tolerance=.0015):
    """
    Fit line to EEG pulse times and behavioral pulse times and apply to event times.

    """
    start_ind = None
    # get starting ind for the beh_times
    for i in range(len(beh_times)//2):
        if np.abs(np.diff(eeg_times[:blen]) - 
                  np.diff(beh_times[i:blen+i])).sum()<(tolerance*blen):
            start_ind = i
            break
    if start_ind is None:
        raise ValueError('No starting point found')

    # iterate, makeing sure each diff is within tolerance
    etimes = []
    btimes = []
    j = 0
    for i,bt in enumerate(beh_times[start_ind:]):
        if (i == 0) or (np.abs((bt-btimes[-1])-(eeg_times[j]-etimes[-1]))<(tolerance)):
            # looks good, so append
            etimes.append(eeg_times[j])
            btimes.append(bt)
            # increment eeg times counter
            j += 1
            #print i,
        else:
            # no good, so say we're skipping
            print '.', #(np.abs((bt-btimes[-1])-(eeg_times[j]-etimes[-1]))),
    print
    # convert to arrays
    etimes = np.array(etimes)
    btimes = np.array(btimes)
    print "Num. matching: ", len(etimes) #,len(btimes)
    #plot(etimes,btimes,'o')

    # fit a line to convert between behavioral and eeg times
    A = np.vstack([btimes, np.ones(len(btimes))]).T
    m, c = np.linalg.lstsq(A, etimes)[0]
    print "Slope and Offset: ", m ,c

    # convert to get eoffsets
    eoffsets = ev_times*m + c

    return eoffsets


def load_pyepl_eeg_pulses(logfile, event_label='UP'):
    """
    Load and process the default eeg log file from PyEPL.  This will
    extract only when the pulses turned on (not off), which is what is
    usually saved by EEG systems.
    """
    # open and read each line
    reader = csv.reader(open(logfile,'rU'),dialect=csv.excel_tab)
    pulses = []
    for row in reader:
        if row[2] == event_label:
            pulses.append(long(row[0]))
    return np.asarray(pulses)

def find_needle_in_haystack(needle, haystack, maxdiff):
    """
    Look for a matching subsequence in a long sequence.
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

def times_to_offsets_old(eeg_times, eeg_offsets, beh_times,
                         samplerate, window=100, thresh_ms=10):
    """
    Fit a line to the eeg times to offsets conversion and then apply
    it to the provided behavioral event times.
    """
    pulse_ms = eeg_times
    annot_ms = eeg_offsets
    
    # pick beginning and end (needle in haystack)
    s_ind = None
    e_ind = None
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
    #samplerate = w.samplerate
    #offsets = np.int64(np.round((m*beh_times + c)*samplerate/1000.))

    # return seconds
    offsets = (m*beh_times + c)/1000.

    return offsets

    
def align_pyepl(wrappedfile, eeglog, events, annot_id='S255', 
                channel_for_sr=0, 
                window=100, thresh_ms=10,
                event_time_id='event_time', eeg_event_label='UP'):
    """
    Take an Events instance and add esrc and eoffset, aligning the
    events to the data in the supplied wrapped file (i.e., you must
    wrap your data with something like EDFWrapper or HDF5Wrapper.)
    This extracts the pulse information from the data's annotations
    and matches it up with the pyepl eeg.eeglog file passed in.

    It returns the updated Events.
    """

    if(not isinstance(wrappedfile,BaseWrapper)):
        raise ValueError('BaseWrapper instance required!')
    
    # point to wrapper
    w = wrappedfile

    # load clean pyepl eeg log
    pulse_ms = load_pyepl_eeg_pulses(eeglog, event_label=eeg_event_label)

    # load annotations from edf
    annot = w.annotations

    # convert seconds to ms for annot_ms
    annot_ms = annot[annot['annotations']==annot_id]['onsets'] * 1000

    # get the offsets
    offsets = times_to_offsets(pulse_ms, annot_ms, events[event_time_id],
                               w.samplerate, window=window, thresh_ms=thresh_ms)

    # add esrc and eoffset to the Events instance
    events = events.add_fields(esrc=np.repeat(w,len(events)),
                               eoffset=offsets)

    # return the updated events
    return events

