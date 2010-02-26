import numpy as np
cimport numpy as np

# set up the types
dtype_f64 = np.float64
ctypedef np.float64_t dtype_f64_t

# handle the externs
cdef extern from "edfwrap.h":
    int read_samples_from_file(char *filepath, 
                               int edfsignal, 
                               long offset,
                               int n, 
                               double *buf)

# cdef extern from "edflib.h":
    # struct edf_hdr_struct:
    #       int       handle
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
          
    # int edfopen_file_readonly(char *path,
    #                           struct edf_hdr_struct *edfhdr,
    #                           int read_annotations)

def read_samples(char *filepath, int edfsignal, long offset, int n):

    # allocate space
    cdef np.ndarray[dtype_f64_t, ndim=1] buf = np.empty((n),dtype=dtype_f64)

    # read samples into buffer
    cdef int nread = read_samples_from_file(filepath,
                                            edfsignal,
                                            offset,
                                            n,
                                            <dtype_f64_t*>buf.data)

    if nread < 0:
        # we had an error, so return none
        nread = 0
    
    # return the buffer, truncated to the number of samples
    return buf[0:nread]

