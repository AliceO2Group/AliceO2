// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file pthread_mutex_win32_wrapper.h
/// \author David Rohr

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

static inline int pthread_mutex_init(pthread_mutex_t* mutex, const void* attr)
{
  *mutex = CreateSemaphore(nullptr, 1, 1, nullptr);
  // printf("INIT %d\n", *mutex);
  return ((*mutex) == nullptr);
}

static inline int pthread_mutex_lock(pthread_mutex_t* mutex)
{
  // printf("LOCK %d\n", *mutex);
  return (WaitForSingleObject(*mutex, INFINITE) == WAIT_FAILED);
}

static inline int pthread_mutex_trylock(pthread_mutex_t* mutex)
{
  DWORD retVal = WaitForSingleObject(*mutex, 0);
  if (retVal == WAIT_TIMEOUT) {
    return (EBUSY);
  }
  // printf("TRYLOCK %d\n", *mutex);
  if (retVal != WAIT_FAILED) {
    return (0);
  }
  return (1);
}

static inline int pthread_mutex_unlock(pthread_mutex_t* mutex)
{
  // printf("UNLOCK %d\n", *mutex);
  return (ReleaseSemaphore(*mutex, 1, nullptr) == 0);
}

static inline int pthread_mutex_destroy(pthread_mutex_t* mutex) { return (CloseHandle(*mutex) == 0); }

static inline int pthread_create(pthread_t* thread, const void* attr, void* (*start_routine)(void*), void* arg) { return ((*thread = CreateThread(nullptr, 0, (LPTHREAD_START_ROUTINE)start_routine, arg, 0, nullptr)) == 0); }

static inline int pthread_exit(void* ret) { ExitThread((DWORD)(size_t)ret); }

static inline int pthread_join(pthread_t thread, void** retval)
{
  static DWORD ExitCode;
  while (GetExitCodeThread(thread, &ExitCode) == STILL_ACTIVE) {
    Sleep(0);
  }
  if (retval != nullptr) {
    *retval = (void*)&ExitCode;
  }
  return (0);
}

static inline int sem_init(sem_t* sem, int pshared, unsigned int value)
{
  *sem = CreateSemaphore(nullptr, value, 1024, nullptr);
  return ((*sem) == nullptr);
}

static inline int sem_destroy(sem_t* sem) { return (CloseHandle(*sem) == 0); }

static inline int sem_wait(sem_t* sem) { return (WaitForSingleObject(*sem, INFINITE) == WAIT_FAILED); }

static inline int sem_trywait(sem_t* sem)
{
  DWORD retVal = WaitForSingleObject(*sem, 0);
  if (retVal == WAIT_TIMEOUT) {
    return (EAGAIN);
  }
  if (retVal != WAIT_FAILED) {
    return (0);
  }
  return (-1);
}

static inline int sem_post(sem_t* sem) { return (ReleaseSemaphore(*sem, 1, nullptr) == 0); }

#ifdef CMODULES_PTHREAD_BARRIERS

typedef SYNCHRONIZATION_BARRIER pthread_barrier_t;

static inline int pthread_barrier_destroy(pthread_barrier_t* b) { return (DeleteSynchronizationBarrier(b) == 0); }

static inline int pthread_barrier_init(pthread_barrier_t* b, void* attr, unsigned count) { return (InitializeSynchronizationBarrier(b, count, -1) == 0); }

static inline int pthread_barrier_wait(pthread_barrier_t* b)
{
  EnterSynchronizationBarrier(b, 0);
  return (0);
}

#endif

#endif
