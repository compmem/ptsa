
#ifndef __EDFWRAP_H__
#define __EDFWRAP_H__

#include "edflib.h"

int open_file_readonly(const char *filepath,
		       struct edf_hdr_struct *hdr,
		       int read_annot);

long long get_samples_in_file(struct edf_hdr_struct *hdr,
			      int edfsignal);

double get_samplerate(struct edf_hdr_struct *hdr,
		      int edfsignal);

int read_samples_from_file(struct edf_hdr_struct *hdr,
			   int edfsignal, 
			   long long offset,
			   int n, 
			   double *buf);

#endif
