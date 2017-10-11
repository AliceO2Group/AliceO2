// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CONCURRENT_QUEUE_H_
#define ALICEO2_CONCURRENT_QUEUE_H_

#include <cassert>
#include <deque>
#include <mutex>
#include <condition_variable>

namespace o2 {
namespace DataDistribution {

namespace impl {

/// Concurrent (thread-safe) container adapter for FIFO/LIFO data structures
enum QueueType {
  eFIFO,
  eLIFO
};

template <typename T, QueueType type>
class ConcurrentContainerImpl {
public:
  ~ConcurrentContainerImpl()
  {
    stop();
  }

  void stop()
  {
    std::unique_lock<std::mutex> lLock(mLock);
    mRunning = false;
    lLock.unlock();
    mCond.notify_all();
  }

  void flush()
  {
    std::unique_lock<std::mutex> lLock(mLock);
    mContainer.clear();
    lLock.unlock();
    mCond.notify_all();
  }

  template <typename... Args>
  void push(Args&&... args)
  {
    std::unique_lock<std::mutex> lLock(mLock);

    if (type == eFIFO)
      mContainer.emplace_front(std::forward<Args>(args)...);
    else if (type == eLIFO)
      mContainer.emplace_back(std::forward<Args>(args)...);

    lLock.unlock(); // reduce contention
    mCond.notify_one();
  }

  bool pop(T& d)
  {
    std::unique_lock<std::mutex> lLock(mLock);
    while (mContainer.empty() && mRunning)
      mCond.wait(lLock);

    if (!mRunning)
      return false; // should stop

    assert(!mContainer.empty());
    d = std::move(mContainer.front());
    mContainer.pop_front();
    return true;
  }

  bool try_pop(T& d)
  {
    std::unique_lock<std::mutex> lLock(mLock);
    if (mContainer.empty() || !mRunning)
      return false;

    assert(!mContainer.empty());
    d = std::move(mContainer.front());
    mContainer.pop_front();
    return true;
  }

  std::size_t size() const
  {
    std::unique_lock<std::mutex> lLock(mLock);
    return mContainer.size();
  }

private:
  std::deque<T> mContainer;
  mutable std::mutex mLock;
  std::condition_variable mCond;
  bool mRunning = true;
};

} /* namespace impl*/

// concurent Queue (FIFO)
template <class T>
using ConcurrentFifo = impl::ConcurrentContainerImpl<T, impl::eFIFO>;

// concurent Stack (LIFO)
template <class T>
using ConcurrentLifo = impl::ConcurrentContainerImpl<T, impl::eLIFO>;
}
} /* namespace o2::DataDistribution */

#endif /* ALICEO2_CONCURRENT_QUEUE_H_ */
