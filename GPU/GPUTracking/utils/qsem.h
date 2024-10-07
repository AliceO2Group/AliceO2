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

/// \file qsem.h
/// \author David Rohr

#ifndef QSEM_H
#define QSEM_H

#ifdef _WIN32
#include "pthread_mutex_win32_wrapper.h"
#else
#include <semaphore.h>
#endif

class qSem
{
 public:
  qSem(int32_t num = 1);
  ~qSem();

  int32_t Lock();
  int32_t Unlock();
  int32_t Trylock();
  int32_t Query();

 private:
  int32_t max;
  sem_t sem;
};

class qSignal
{
 private:
  qSem sem;

 public:
  qSignal() : sem(0) {}
  void Wait() { sem.Lock(); }
  void Signal() { sem.Unlock(); }
};

#endif
