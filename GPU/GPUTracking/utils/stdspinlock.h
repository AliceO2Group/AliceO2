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

/// \file stdspinlock.h
/// \author David Rohr

#ifndef Q_STDSPINLOCK_H
#define Q_STDSPINLOCK_H

#include <atomic>

class stdspinlock
{
 public:
  stdspinlock(std::atomic_flag& flag) : mFlag(&flag)
  {
    while (flag.test_and_set(std::memory_order_acquire)) {
    }
  }
  void release()
  {
    if (mFlag) {
      mFlag->clear(std::memory_order_release);
      mFlag = nullptr;
    }
  }
  ~stdspinlock()
  {
    release();
  }

 private:
  std::atomic_flag* mFlag;
};

#endif // Q_STDSPINLOCK_H
