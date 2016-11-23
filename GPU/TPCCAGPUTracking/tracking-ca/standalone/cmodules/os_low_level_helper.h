#ifndef OS_LOW_LEVEL_HELPER_H
#define OS_LOW_LEVEL_HELPER_H

#ifndef _WIN32
#include <syscall.h>
#include <unistd.h> 
#endif

inline int get_number_of_cpu_cores()
{
#ifdef _WIN32
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	return(info.dwNumberOfProcessors);
#else
	return(sysconf(_SC_NPROCESSORS_ONLN));
#endif
}

inline int get_standard_page_size()
{
#ifdef _WIN32
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	return(info.dwPageSize);
#else
	return(getpagesize());
#endif
}

#endif
