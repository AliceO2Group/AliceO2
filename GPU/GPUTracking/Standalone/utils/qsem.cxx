// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file qsem.cxx
/// \author David Rohr

#include <cerrno>
#include <cstdio>

#include "qsem.h"

#ifndef STD_OUT
#define STD_OUT stdout
#endif

qSem::qSem(int num)
{
  max = num;
  if (sem_init(&sem, 0, num)) {
    fprintf(STD_OUT, "Error initializing semaphore");
  }
}

qSem::~qSem()
{
  if (sem_destroy(&sem)) {
    fprintf(STD_OUT, "Error destroying semaphore");
  }
}

int qSem::Lock()
{
  int retVal;
  if ((retVal = sem_wait(&sem))) {
    fprintf(STD_OUT, "Error locking semaphore");
  }
  return (retVal);
}

int qSem::Unlock()
{
  int retVal;
  if ((retVal = sem_post(&sem))) {
    fprintf(STD_OUT, "Error unlocking semaphire");
  }
  return (retVal);
}

int qSem::Trylock()
{
  int retVal = sem_trywait(&sem);
  if (retVal) {
    if (errno == EAGAIN) {
      return (EBUSY);
    }
    return (-1);
  }
  return (0);
}

#ifndef _WIN32
int qSem::Query()
{
  int value;
  if (sem_getvalue(&sem, &value) != 0) {
    value = -1;
  }
  return (value);
}
#endif
