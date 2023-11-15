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

/// \author ruben.shahoyan@cern.ch
/// Thread-safe FIFO

#ifndef ALICEO2_FIFOUTILS_H_
#define ALICEO2_FIFOUTILS_H_

#include <deque>
#include <mutex>
#include <stdexcept>

namespace o2
{
namespace utils
{

template <typename T>
class FIFO
{
 public:
  size_t size() const
  {
    std::lock_guard<std::mutex> lock(mMutex);
    return mQueue.size();
  }

  void clear()
  {
    std::lock_guard<std::mutex> lock(mMutex);
    mQueue.clear();
  }

  bool empty() const
  {
    std::lock_guard<std::mutex> lock(mMutex);
    return mQueue.empty();
  }

  template <typename... Args>
  void push(Args&&... args)
  {
    std::lock_guard<std::mutex> lock(mMutex);
    mQueue.emplace_back(std::forward<Args>(args)...);
  }

  void pop()
  {
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mQueue.empty()) {
      mQueue.pop_front();
    }
  }

  const T& front() const
  {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mQueue.empty()) {
      throw std::runtime_error("attempt to access front of empty queue");
    }
    return mQueue.front();
  }

  T& front()
  {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mQueue.empty()) {
      throw std::runtime_error("attempt to access front of empty queue");
    }
    return mQueue.front();
  }

  const T* frontPtr() const
  {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mQueue.empty()) {
      return nullptr;
    }
    return &mQueue.front();
  }

  T* frontPtr()
  {
    std::lock_guard<std::mutex> lock(mMutex);
    if (mQueue.empty()) {
      return nullptr;
    }
    return &mQueue.front();
  }

  auto& getQueue() const { return mQueue; }

 private:
  mutable std::mutex mMutex;
  std::deque<T> mQueue{};
};

} // namespace utils
} // namespace o2

#endif
