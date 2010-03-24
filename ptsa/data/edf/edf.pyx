import numpy as np
cimport numpy as np

# set up the types
dtype_f64 = np.float64
ctypedef np.float64_t dtype_f64_t

# handle the externs
cdef extern from "edflib.h":
    struct edf_hdr_struct:
        int       handle
        long annotations_in_file

    # Dummy enums for variables defined by macros in the header
    enum:
        EDFLIB_TIME_DIMENSION
        EDFLIB_DO_NOT_READ_ANNOTATIONS
        EDFLIB_READ_ANNOTATIONS
        EDFLIB_READ_ALL_ANNOTATIONS
        EDFLIB_MAX_ANNOTATION_LEN

    struct edf_annotation_struct:
        long onset
        char *duration
        char *annotation

    int edf_get_annotation(int handle, int n, edf_annotation_struct *annot)
    
    int edfclose_file(int handle)


cdef extern from "edfwrap.h":
    int open_file_readonly(char *filepath,
                           edf_hdr_struct *hdr,
                           int read_annot)
    float get_samplerate(edf_hdr_struct *hdr,
                         int edfsignal)
    int read_samples_from_file(edf_hdr_struct *hdr,
                               int edfsignal, 
                               long offset,
                               int n, 
                               double *buf)

def read_annotations(char *filepath):
    """
    read_annotations(filepath)

    Read in all the annotations from an EDF/BDF file into a record
    array. Note that the onset times converted to seconds.

    Parameters
    ----------
    filepath : {str}
        The path and name of the EDF/BDF file.

    Returns
    -------
    annotations : {np.recarray}
        A record array with onsets, duration, and annotations.
    
    """
    # get a header
    cdef edf_hdr_struct hdr

    # open the file
    if open_file_readonly(filepath, &hdr, EDFLIB_READ_ALL_ANNOTATIONS) < 0:
        print "Error opening file."
        return None

    # allocate for an annotation
    cdef edf_annotation_struct annot

    # this could be improved
    cdef np.ndarray[dtype_f64_t, ndim=1] onsets = np.empty(
        hdr.annotations_in_file,dtype=dtype_f64)
    durations = []
    annotations = []
    
    # loop over annotations
    for i in range(hdr.annotations_in_file):
        if edf_get_annotation(hdr.handle, i, &annot):
            print "Error reading annotation %d" % (i)
            return None

        # append the annotations
        onsets[i] = annot.onset
        durations.append(annot.duration)
        annotations.append(annot.annotation)

    # close the file
    edfclose_file(hdr.handle)

    # return record array of annotations
    return np.rec.fromarrays(
        [onsets/EDFLIB_TIME_DIMENSION,durations,annotations],
        names='onsets,durations,annotations')
        
def read_samplerate(char *filepath, int edfsignal):
    """
    read_samplerate(filepath, edfsignal)

    Read the samplerate for a signal in an EDF/BDF file.  Note that
    different signals can have different samplerates.

    Parameters
    ----------
    filepath : {str}
        The path and name of the EDF/BDF file.
    edfsignal : {int}
        The signal whose samplerate to retrieve.
        
    Returns
    -------
    samplerate : {float}
        The samplerate for that signal.

    """
    # get a header
    cdef edf_hdr_struct hdr

    # open the file
    if open_file_readonly(filepath, &hdr, EDFLIB_DO_NOT_READ_ANNOTATIONS) < 0:
        print "Error opening file."
        return None

    # get the samplerate
    cdef float samplerate = get_samplerate(&hdr,
                                           edfsignal)

    # close the file
    edfclose_file(hdr.handle)

    return samplerate

def read_samples(char *filepath, int edfsignal, long offset, int n):
    """
    read_samples(filepath, edfsignal, offset, n)

    Read in samples from a signal in an EDF/BDF file.

    Parameters
    ----------
    filepath : {str}
        The path and name of the EDF/BDF file.
    edfsignal : {int}
        The signal whose samplerate to retrieve.
    offset : {long}
        Offset in samples into the file where to start reading.
    n : {int}
        Number of samples to read, starting at offset.
        
    Returns
    -------
    samples : {np.ndarray}
        An ndarray of samples read from the file.

    """
    # allocate space
    cdef np.ndarray[dtype_f64_t, ndim=1] buf = np.empty((n),dtype=dtype_f64)

    # get a header
    cdef edf_hdr_struct hdr

    # open the file
    if open_file_readonly(filepath, &hdr, EDFLIB_DO_NOT_READ_ANNOTATIONS) < 0:
        print "Error opening file."
        return None
    
    # read samples into buffer
    cdef int nread = read_samples_from_file(&hdr,
                                            edfsignal,
                                            offset,
                                            n,
                                            <dtype_f64_t*>buf.data)

    if nread < 0:
        # we had an error, so return none
        nread = 0

    # close the file
    edfclose_file(hdr.handle)
    
    # return the buffer, truncated to the number of samples
    return buf[0:nread]

