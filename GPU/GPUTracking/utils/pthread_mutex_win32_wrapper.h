#ifndef PTHREAD_MUTEX_WIN32_WRAPPER_H
#define PTHREAD_MUTEX_WIN32_WRAPPER_H

#include <windows.h>
#include <winbase.h>
typedef HANDLE pthread_mutex_t;
typedef HANDLE pthread_t;
typedef HANDLE sem_t;

#ifndef EBUSY
#define EBUSY WAIT_TIMEOUT
#endif

#ifndef EAGAIN
#define EAGAIN WAIT_TIMEOUT
#endif

static inline int pthread_mutex_init(pthread_mutex_t *mutex, const void* attr)
{
	*mutex = CreateSemaphore(NULL, 1, 1, NULL);
	//printf("INIT %d\n", *mutex);
	return((*mutex) == NULL);
}

static inline int pthread_mutex_lock(pthread_mutex_t *mutex)
{
	//printf("LOCK %d\n", *mutex);
	return(WaitForSingleObject(*mutex, INFINITE) == WAIT_FAILED);
}

static inline int pthread_mutex_trylock(pthread_mutex_t *mutex)
{
	DWORD retVal = WaitForSingleObject(*mutex, 0);
	if (retVal == WAIT_TIMEOUT) return(EBUSY);
	//printf("TRYLOCK %d\n", *mutex);
	if (retVal != WAIT_FAILED) return(0);
	return(1);
}

static inline int pthread_mutex_unlock(pthread_mutex_t *mutex)
{
	//printf("UNLOCK %d\n", *mutex);
	return(ReleaseSemaphore(*mutex, 1, NULL) == 0);
}

static inline int pthread_mutex_destroy(pthread_mutex_t *mutex)
{
	return(CloseHandle(*mutex) == 0);
}

static inline int pthread_create(pthread_t *thread, const void* attr, void *(*start_routine)(void*), void *arg)
{
	return((*thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) start_routine, arg, 0, NULL)) == 0);
}

static inline int pthread_exit(void* ret)
{
	ExitThread((DWORD) (size_t) ret);
}

static inline int pthread_join(pthread_t thread, void** retval)
{
	static DWORD ExitCode;
	while (GetExitCodeThread(thread, &ExitCode) == STILL_ACTIVE) Sleep(0);
	if (retval != NULL) *retval = (void*) &ExitCode;
	return(0);
}

static inline int sem_init(sem_t *sem, int pshared, unsigned int value)
{
	*sem = CreateSemaphore(NULL, value, 1024, NULL);
	return((*sem) == NULL);
}

static inline int sem_destroy(sem_t *sem)
{
	return(CloseHandle(*sem) == 0);
}

static inline int sem_wait(sem_t *sem)
{
	return(WaitForSingleObject(*sem, INFINITE) == WAIT_FAILED);
}

static inline int sem_trywait(sem_t *sem)
{
	DWORD retVal = WaitForSingleObject(*sem, 0);
	if (retVal == WAIT_TIMEOUT) return(EAGAIN);
	if (retVal != WAIT_FAILED) return(0);
	return(-1);
}

static inline int sem_post(sem_t *sem)
{
	return(ReleaseSemaphore(*sem, 1, NULL) == 0);
}

#ifdef CMODULES_PTHREAD_BARRIERS

/*typedef struct _RTL_BARRIER {                       
            DWORD Reserved1;                        
            DWORD Reserved2;                        
            ULONG_PTR Reserved3[2];                 
            DWORD Reserved4;                        
            DWORD Reserved5;                        
} RTL_BARRIER, *PRTL_BARRIER;  

typedef RTL_BARRIER SYNCHRONIZATION_BARRIER;
typedef PRTL_BARRIER PSYNCHRONIZATION_BARRIER;
typedef PRTL_BARRIER LPSYNCHRONIZATION_BARRIER;

#define SYNCHRONIZATION_BARRIER_FLAGS_SPIN_ONLY  0x01
#define SYNCHRONIZATION_BARRIER_FLAGS_BLOCK_ONLY 0x02
#define SYNCHRONIZATION_BARRIER_FLAGS_NO_DELETE  0x04

BOOL WINAPI EnterSynchronizationBarrier(_Inout_ LPSYNCHRONIZATION_BARRIER lpBarrier, _In_ DWORD dwFlags);
BOOL WINAPI InitializeSynchronizationBarrier(_Out_ LPSYNCHRONIZATION_BARRIER lpBarrier, _In_ LONG lTotalThreads, _In_ LONG lSpinCount);
BOOL WINAPI DeleteSynchronizationBarrier(_Inout_ LPSYNCHRONIZATION_BARRIER lpBarrier);*/

typedef SYNCHRONIZATION_BARRIER pthread_barrier_t;

static inline int pthread_barrier_destroy(pthread_barrier_t* b)
{
	return(DeleteSynchronizationBarrier(b) == 0);
}

static inline int pthread_barrier_init(pthread_barrier_t* b, void* attr, unsigned count)
{
	return(InitializeSynchronizationBarrier(b, count, -1) == 0);
}

static inline int pthread_barrier_wait(pthread_barrier_t* b)
{
	EnterSynchronizationBarrier(b, 0);
	return(0);
}

#endif

#endif
