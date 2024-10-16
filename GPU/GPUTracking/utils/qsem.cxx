// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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

qSem::qSem(int32_t num)
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

int32_t qSem::Lock()
{
  int32_t retVal;
  if ((retVal = sem_wait(&sem))) {
    fprintf(STD_OUT, "Error locking semaphore");
  }
  return (retVal);
}

int32_t qSem::Unlock()
{
  int32_t retVal;
  if ((retVal = sem_post(&sem))) {
    fprintf(STD_OUT, "Error unlocking semaphire");
  }
  return (retVal);
}

int32_t qSem::Trylock()
{
  int32_t retVal = sem_trywait(&sem);
  if (retVal) {
    if (errno == EAGAIN) {
      return (EBUSY);
    }
    return (-1);
  }
  return (0);
}

#ifndef _WIN32
int32_t qSem::Query()
{
  int32_t value;
  if (sem_getvalue(&sem, &value) != 0) {
    value = -1;
  }
  return (value);
}
#endif
