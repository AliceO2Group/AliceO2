// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  qSem(int num = 1);
  ~qSem();

  int Lock();
  int Unlock();
  int Trylock();
  int Query();

 private:
  int max;
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
