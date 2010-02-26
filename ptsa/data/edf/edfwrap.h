

#ifndef __EDFWRAP_H__
#define __EDFWRAP_H__

int read_samples_from_file(const char *filepath, 
			   int edfsignal, 
			   long long offset,
			   int n, 
			   double *buf);

#endif
