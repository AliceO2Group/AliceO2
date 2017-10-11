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
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <iterator>

namespace o2
{
namespace DataDistribution
{

namespace impl
{

/// Concurrent (thread-safe) container adapter for FIFO/LIFO data structures
enum QueueType {
  eFIFO,
  eLIFO
};

template <typename T, QueueType type>
class ConcurrentContainerImpl
{
 public:
  typedef T value_type;

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

    if (type == eFIFO) {
      mContainer.emplace_back(std::forward<Args>(args)...);
    } else if (type == eLIFO) {
      mContainer.emplace_front(std::forward<Args>(args)...);
    }

    lLock.unlock(); // reduce contention
    mCond.notify_one();
  }

  bool pop(T& d)
  {
    std::unique_lock<std::mutex> lLock(mLock);
    while (mContainer.empty() && mRunning) {
      mCond.wait(lLock);
    }

    if (!mRunning && mContainer.empty())
      return false;

    assert(!mContainer.empty());
    d = std::move(mContainer.front());
    mContainer.pop_front();
    return true;
  }

  template <class OutputIt>
  unsigned long pop_n(const unsigned long pCnt, OutputIt pDstIter)
  {
    std::unique_lock<std::mutex> lLock(mLock);
    while (mContainer.empty() && mRunning) {
      mCond.wait(lLock);
    }

    if (!mRunning && mContainer.empty())
      return false; // should stop

    assert(!mContainer.empty());

    unsigned long ret = std::min(mContainer.size(), pCnt);
    std::move(std::begin(mContainer), std::begin(mContainer) + ret, pDstIter);
    mContainer.erase(std::begin(mContainer), std::begin(mContainer) + ret);
    return ret;
  }

  bool try_pop(T& d)
  {
    std::unique_lock<std::mutex> lLock(mLock);
    if (mContainer.empty()) {
      return false;
    }

    d = std::move(mContainer.front());
    mContainer.pop_front();
    return true;
  }

  template <class OutputIt>
  unsigned long try_pop_n(const unsigned long pCnt, OutputIt pDstIter)
  {
    std::unique_lock<std::mutex> lLock(mLock);
    if (mContainer.empty()) {
      return 0;
    }

    unsigned long ret = std::min(mContainer.size(), pCnt);
    std::move(std::begin(mContainer), std::begin(mContainer) + ret, pDstIter);
    mContainer.erase(std::begin(mContainer), std::begin(mContainer) + ret);
    return ret;
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

///
///  ConcurrentContainerImpl specializations for o2::DataDistribution
///

// concurent Queue (FIFO)
template <class T>
using ConcurrentFifo = impl::ConcurrentContainerImpl<T, impl::eFIFO>;

// concurent Stack (LIFO)
template <class T>
using ConcurrentLifo = impl::ConcurrentContainerImpl<T, impl::eLIFO>;
}

///
///  Interface for an object with input and output ConcurrentContainer queue/stack
///

template <
  typename T,
  typename = std::enable_if_t<std::is_move_assignable<T>::value>>
class IFifoPipeline
{

 public:
  IFifoPipeline(unsigned pNoStages)
    : mPipelineQueues(pNoStages)
  {
  }

  virtual ~IFifoPipeline() {}

  void stopPipeline()
  {
    for (auto& lQueue : mPipelineQueues) {
      lQueue.stop();
    }
  }

  template <typename... Args>
  bool queue(unsigned pStage, Args&&... args)
  {
    assert(pStage < mPipelineQueues.size());
    auto lNextStage = getNextPipelineStage(pStage);
    assert((lNextStage <= mPipelineQueues.size()) && "next stage larger than expected");
    // NOTE: (lNextStage == mPipelineQueues.size()) is the drop queue
    if (lNextStage < mPipelineQueues.size()) {
      mPipelineQueues[lNextStage].push(std::forward<Args>(args)...);
      return true;
    }
    return false;
  }

  T dequeue(unsigned pStage)
  {
    T t;
    mPipelineQueues[pStage].pop(t);
    return std::move(t);
  }

 protected:
  virtual unsigned getNextPipelineStage(unsigned pStage) = 0;

  std::vector<o2::DataDistribution::ConcurrentFifo<T>> mPipelineQueues;
};

} /* namespace o2::DataDistribution */

#endif /* ALICEO2_CONCURRENT_QUEUE_H_ */
