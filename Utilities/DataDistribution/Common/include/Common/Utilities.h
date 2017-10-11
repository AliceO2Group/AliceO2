// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DATADIST_UTILITIES_H_
#define ALICEO2_DATADIST_UTILITIES_H_

#include <O2Device/O2Device.h>

#include <type_traits>
#include <memory>

#include <vector>

namespace o2 {
namespace DataDistribution {
using namespace o2::Base;

static constexpr std::uintptr_t gChanPtrAlign = 32;

class ChannelAllocator;

template <class T>
class ChannelPtr {
  friend class ChannelAllocator;

public:
  typedef T element_type;
  typedef T value_type;
  typedef T* pointer;
  typedef std::ptrdiff_t difference_type;
  typedef typename std::add_lvalue_reference<element_type>::type reference;

  ChannelPtr() = default;
  ~ChannelPtr() = default;

  explicit ChannelPtr(const ChannelPtr&) = delete;
  ChannelPtr& operator=(const ChannelPtr&) = delete;

  ChannelPtr(ChannelPtr&&) = default;
  ChannelPtr& operator=(ChannelPtr&& a) = default;

  ChannelPtr& operator=(FairMQMessagePtr&& a) noexcept
  {
    assert(a->GetSize() == sizeof(T) + gChanPtrAlign - 1);
    mMessage = std::move(a);
    return *this;
  }

  void reset(ChannelPtr& p)
  {
    mMessage = std::move(p);
  }
  void swap(ChannelPtr& p) noexcept
  {
    mMessage.swap(p.mMessage);
  }

  pointer get() const noexcept
  {
    return reinterpret_cast<pointer>((reinterpret_cast<std::uintptr_t>(mMessage->GetData()) + gChanPtrAlign - 1) /
                                     gChanPtrAlign * gChanPtrAlign);
  }

  reference operator*() const noexcept
  {
    assert(get() != nullptr);
    return *get();
  }

  pointer operator->() const noexcept
  {
    assert(get() != nullptr);
    return get();
  }

  explicit operator bool() const noexcept
  {
    return mMessage != nullptr;
  }

  FairMQMessagePtr get_message()
  {
    return std::move(mMessage);
  }

private:
  ChannelPtr(FairMQMessagePtr& m) : mMessage(std::move(m))
  {
  }
  FairMQMessagePtr mMessage;
};

class ChannelAllocator {
public:
  static ChannelAllocator& get()
  {
    static ChannelAllocator sInstance;
    return sInstance;
  }

  struct ChannelAllocatorPrivate {
    O2Device* mDevice;
    std::string mChannelName;
    int mChannelIdx;
  };

  void addChannel(const int pChannId, O2Device* pDev, const std::string& pChanName, const int pChanIdx)
  {
    mChannels[pChannId] = ChannelAllocatorPrivate{ pDev, pChanName, pChanIdx };
  }

  template <class T>
  ChannelPtr<T> allocate(const int pChannId)
  {
    assert(mChannels.count(pChannId) == 1);
    ChannelAllocatorPrivate& lChanPriv = mChannels.at(pChannId);
    FairMQMessagePtr lMessage = std::move(
      lChanPriv.mDevice->NewMessageFor(lChanPriv.mChannelName, lChanPriv.mChannelIdx, sizeof(T) + gChanPtrAlign - 1));

    ChannelPtr<T> lChanPtr(lMessage);

    std::memset(lChanPtr.get(), 0, sizeof(T));
    return lChanPtr;
  }

private:
  ChannelAllocator() = default;
  ChannelAllocator(ChannelAllocator&) = delete;
  ChannelAllocator operator=(ChannelAllocator&) = delete;

  std::map<int, ChannelAllocatorPrivate> mChannels;
};

template <class T, class... Args>
inline ChannelPtr<T> make_channel_ptr(const int pChannId, Args&&... args)
{
  ChannelPtr<T> lPtr(std::move(ChannelAllocator::get().allocate<T>(pChannId)));
  assert(lPtr.get() != nullptr);
  new (lPtr.get()) T(std::forward<Args>(args)...);
  return lPtr;
}

// TODO: thread safety if widely used
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
class RunningSamples {
public:
  RunningSamples() = delete;
  RunningSamples(std::size_t pCnt, const T pInitVal = T(0)) : mSamples(std::max(pCnt, std::size_t(1)), pInitVal)
  {
  }

  void Fill(const T pNewVal)
  {
    T& lOldVal = mSamples[mIndex];
    mSum = mSum + pNewVal - lOldVal;
    lOldVal = pNewVal;
    mIndex = (mIndex + 1) % mSamples.size();
    mCount = std::min(mCount + 1, mSamples.size());
  }

  T Sum() const
  {
    return mSum;
  }
  double Mean() const
  {
    return double(mSum) / double(mCount);
  }

  auto begin() const
  {
    return mSamples.begin();
  }
  auto end() const
  {
    return mSamples.begin() + mCount;
  }

private:
  std::vector<T> mSamples;
  T mSum = T(0);
  std::size_t mIndex = 0;
  std::size_t mCount = 0;
};
}
} /* namespace o2::DataDistribution */

#endif /* ALICEO2_DATADIST_UTILITIES_H_ */
