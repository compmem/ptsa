import numpy as np
cimport numpy as np

# set up the types
dtype_f64 = np.float64
ctypedef np.float64_t dtype_f64_t

# handle the externs
cdef extern from "edflib.h":
    struct edf_hdr_struct:
        int       handle
    #       int       filetype
    #       int       edfsignals
    #       long      file_duration  # was long long
    #       int       startdate_day
    #       int       startdate_month
    #       int       startdate_year
    #       long      starttime_subsecond # was long long
    #       int       starttime_second
    #       int       starttime_minute
    #       int       starttime_hour

    int edfclose_file(int handle)

    # int edfopen_file_readonly(char *path,
    #                           struct edf_hdr_struct *edfhdr,
    #                           int read_annotations)

cdef extern from "edfwrap.h":
    int open_file_readonly(char *filepath,
                           edf_hdr_struct *hdr)
    float get_samplerate(edf_hdr_struct *hdr,
                         int edfsignal)
    int read_samples_from_file(edf_hdr_struct *hdr,
                               int edfsignal, 
                               long offset,
                               int n, 
                               double *buf)

def read_samplerate(char *filepath, int edfsignal):
    # get a header
    cdef edf_hdr_struct hdr

    # open the file
    if open_file_readonly(filepath, &hdr) < 0:
        print "Error opening file."
        return None

    # get the samplerate
    cdef float samplerate = get_samplerate(&hdr,
                                           edfsignal)

    # close the file
    edfclose_file(hdr.handle)

    return samplerate

def read_samples(char *filepath, int edfsignal, long offset, int n):

    # allocate space
    cdef np.ndarray[dtype_f64_t, ndim=1] buf = np.empty((n),dtype=dtype_f64)

    # get a header
    cdef edf_hdr_struct hdr

    # open the file
    if open_file_readonly(filepath, &hdr) < 0:
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

