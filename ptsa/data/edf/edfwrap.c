
#include <stdio.h>
#include <stdlib.h>

#include "edflib.h"

#include "edfwrap.h"

int read_samples_from_file(const char *filepath, 
			   int edfsignal, 
			   long long offset,
			   int n, 
			   double *buf)
{
  int hdl;
  struct edf_hdr_struct hdr;

  if(edfopen_file_readonly(filepath, &hdr, EDFLIB_DO_NOT_READ_ANNOTATIONS))
  {
    switch(hdr.filetype)
    {
      case EDFLIB_MALLOC_ERROR                : printf("\nmalloc error\n\n");
                                                break;
      case EDFLIB_NO_SUCH_FILE_OR_DIRECTORY   : printf("\ncan not open file, no such file or directory\n\n");
                                                break;
      case EDFLIB_FILE_CONTAINS_FORMAT_ERRORS : printf("\nthe file is not EDF(+) or BDF(+) compliant\n"
                                                       "(it contains format errors)\n\n");
                                                break;
      case EDFLIB_MAXFILES_REACHED            : printf("\nto many files opened\n\n");
                                                break;
      case EDFLIB_FILE_READ_ERROR             : printf("\na read error occurred\n\n");
                                                break;
      case EDFLIB_FILE_ALREADY_OPENED         : printf("\nfile has already been opened\n\n");
                                                break;
      default                                 : printf("\nunknown error\n\n");
                                                break;
    }

    return(-1);
  }

  // get the handle
  hdl = hdr.handle;

  // check the channel
  if(edfsignal>(hdr.edfsignals))
  {
    printf("\nerror: file has %i signals and you selected signal %i\n\n", 
	   hdr.edfsignals, edfsignal);
    edfclose_file(hdl);
    return(-1);
  }

  // seek to the correct point in the file
  edfseek(hdl, edfsignal, offset, SEEK_SET);

  // read the samples
  n = edfread_physical_samples(hdl, edfsignal, n, buf);

  // close the file
  edfclose_file(hdl);
  
  // return how many we read
  return n;
}
