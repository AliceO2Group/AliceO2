// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//-*- Mode: C++ -*-

#ifndef FIFO_H
#define FIFO_H
//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   Fifo.h
//  @author Matthias Richter
//  @since  2016-12-07
//  @brief  Thread safe FIFO

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace o2 {
namespace test {

/**
 * @class Fifo
 * @brief A thread safe FIFO
 *
 * The class inherits protected, this makes all methods of the
 * underlying container inaccessible to the outside. All methods
 * can only be called through the implemented methods of this class
 * which are thread protected.
 *
 * TODO: not sure at the moment what is the best option to indicate
 * end-of-data in the FIFO. Right now, the push method takes an optional
 * parameter to indicate the last insertion. This allows for checking in
 * the pull method whether data can be expected despite of a currently
 * empty queue, or if the pull should be terminated immediately.
 */
template<class T
         , class _BASE = std::queue<T>
         >
class Fifo : protected _BASE {
 public:
  Fifo() : mMutex(), mFillStatus(), mStop(false) {}
  using value_type = T;

  /**
   * Push value to the FIFO
   */
  void push(T something, bool isLast = false) {
    std::lock_guard<std::mutex> lock(mMutex);
    mStop |= isLast;
    _BASE::push(something);
    // notify_one will also work in case of multiple consumers
    mFillStatus.notify_one();
  }

  /**
   * Check if FIFO empty
   *
   * TODO: make this const, but then the const'ness must be cast away
   * in order to lock the mutex
   */
  bool empty() {
    std::lock_guard<std::mutex> lock(mMutex);
    return _BASE::empty();
  }

  /**
   * Pull one value from the FIFO and process it using the provided
   * processor function. The return value from the processor is propagated
   * to indicate whether to continue pulling from the FIFO or not.
   */
  template<typename F>
  bool pull(F processor /*, const std::chrono::milliseconds& timeout*/) {
    T value;
    {
      std::unique_lock<std::mutex> lock(mMutex);
      if (_BASE::empty() && !mStop) {
        // TODO: proper implementation with timeout with wait_for, return
        // value will be std::cv_status::no_timeout. The condition variable
        // is allowed to wake up spuriously, also then the return value is
        // std::cv_status::no_timeout, so the queue needs to be checked again
        mFillStatus.wait(lock);
      }
      value = _BASE::front();
    }
    bool keepGoing = processor(value);
    {
      std::unique_lock<std::mutex> lock(mMutex);
      _BASE::pop();
      if (mStop && _BASE::empty()) {
        keepGoing = false;
        mStop = false;
      }
    }
    return keepGoing;
  }

 private:
  std::mutex mMutex;
  std::condition_variable mFillStatus;
  bool mStop;
};

}; // namespace test
}; // namespace o2
#endif
