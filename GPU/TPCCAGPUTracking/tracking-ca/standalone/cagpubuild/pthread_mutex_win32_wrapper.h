#ifndef PTHREAD_MUTEX_WIN32_WRAPPER_H
#define PTHREAD_MUTEX_WIN32_WRAPPER_H

#include <windows.h>
#include <winbase.h>
typedef HANDLE pthread_mutex_t;
typedef HANDLE pthread_t;

#ifndef EBUSY
#define EBUSY WAIT_TIMEOUT
#endif

inline int pthread_mutex_init(pthread_mutex_t *mutex, const void* attr)
{
	return((*mutex = CreateSemaphore(NULL, 1, 1, NULL)) == NULL);
}

inline int pthread_mutex_lock(pthread_mutex_t *mutex)
{
	return(WaitForSingleObject(*mutex, INFINITE) == WAIT_FAILED);
}

inline int pthread_mutex_trylock(pthread_mutex_t *mutex)
{
	DWORD retVal = WaitForSingleObject(*mutex, 0);
	if (retVal == WAIT_OBJECT_0) return(0);
	if (retVal == WAIT_TIMEOUT) return(EBUSY);
	return(1);
}

inline int pthread_mutex_unlock(pthread_mutex_t *mutex)
{
	return(ReleaseSemaphore(*mutex, 0, NULL) == 0);
}

inline int pthread_mutex_destroy(pthread_mutex_t *mutex)
{
	return(CloseHandle(*mutex) == 0);
}

inline int pthread_create(pthread_t *thread, const void* attr, void *(*start_routine)(void*), void *arg)
{
	return((*thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) start_routine, arg, 0, NULL)) == 0);
}

#endif