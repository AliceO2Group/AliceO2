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
#ifndef O2_FRAMEWORK_INPUTSPAN_H_
#define O2_FRAMEWORK_INPUTSPAN_H_

#include "Framework/DataRef.h"
#include <functional>

extern template class std::function<o2::framework::DataRef(size_t, o2::framework::DataRefIndices)>;
extern template class std::function<o2::framework::DataRefIndices(size_t, o2::framework::DataRefIndices)>;

namespace o2::framework
{

/// Mapping helper between the store of all inputs being processed and the
/// actual inputs to be processed in a given go.
/// In general this will use an helper which returns
/// `fair::mq::Message->GetData()` from the Message cache, but in principle
/// the mechanism should be flexible enough to allow all kind of input stores.
class InputSpan
{
 public:
  InputSpan() = delete;
  InputSpan(InputSpan const&) = delete;
  InputSpan(InputSpan&&) = default;

  /// Navigate the message store via the DataRefIndices protocol.
  /// get_next_pair (DataModelViews.h) provides O(1) sequential advancement for nextIndicesGetter.
  InputSpan(std::function<size_t(size_t)> nofPartsGetter,
            std::function<int(size_t)> refCountGetter,
            std::function<DataRef(size_t, DataRefIndices)> indicesGetter,
            std::function<DataRefIndices(size_t, DataRefIndices)> nextIndicesGetter,
            size_t size);

  /// @a i-th element of the InputSpan (O(partidx) sequential scan via indices protocol)
  [[nodiscard]] DataRef get(size_t i, size_t partidx = 0) const
  {
    DataRefIndices idx{0, 1};
    for (size_t p = 0; p < partidx; ++p) {
      idx = mNextIndicesGetter(i, idx);
    }
    return mIndicesGetter(i, idx);
  }

  /// Return the DataRef for the part described by @a indices in slot @a slotIdx in O(1).
  [[nodiscard]] DataRef getAtIndices(size_t slotIdx, DataRefIndices indices) const
  {
    return mIndicesGetter(slotIdx, indices);
  }

  /// Advance from @a current to the indices of the next part in slot @a slotIdx in O(1).
  [[nodiscard]] DataRefIndices nextIndices(size_t slotIdx, DataRefIndices current) const
  {
    return mNextIndicesGetter(slotIdx, current);
  }

  // --- slot-level Iterator protocol (headerIdx doubles as slot position) ---
  [[nodiscard]] DataRefIndices initialIndices() const { return {0, 0}; }
  [[nodiscard]] DataRefIndices endIndices() const { return {mSize, 0}; }
  [[nodiscard]] DataRef getAtIndices(DataRefIndices indices) const { return mIndicesGetter(indices.headerIdx, {0, 1}); }
  [[nodiscard]] DataRefIndices nextIndices(DataRefIndices current) const { return {current.headerIdx + 1, 0}; }

  /// @a number of parts in the i-th element of the InputSpan
  [[nodiscard]] size_t getNofParts(size_t i) const
  {
    if (i >= mSize) {
      return 0;
    }
    return mNofPartsGetter(i);
  }

  // Get the refcount for a given part
  [[nodiscard]] int getRefCount(size_t i) const
  {
    if (i >= mSize) {
      return 0;
    }
    if (!mRefCountGetter) {
      return -1;
    }
    return mRefCountGetter(i);
  }

  /// Number of elements in the InputSpan
  [[nodiscard]] size_t size() const
  {
    return mSize;
  }

  [[nodiscard]] const char* header(size_t i) const
  {
    return get(i).header;
  }

  [[nodiscard]] const char* payload(size_t i) const
  {
    return get(i).payload;
  }

  /// An iterator over the elements of a parent container using the DataRefIndices protocol.
  /// ParentT must provide: initialIndices(), getAtIndices(DataRefIndices), nextIndices(DataRefIndices).
  template <typename ParentT, typename T>
  class Iterator
  {
   public:
    using ParentType = ParentT;
    using SelfType = Iterator;
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using reference = T&;
    using pointer = T*;
    using difference_type = std::ptrdiff_t;
    using ElementType = typename std::remove_const<value_type>::type;

    Iterator() = delete;

    Iterator(ParentType const* parent, bool isEnd = false)
      : mParent(parent),
        mCurrentIndices(isEnd ? parent->endIndices() : parent->initialIndices()),
        mElement{}
    {
      if (mCurrentIndices != mParent->endIndices()) {
        mElement = mParent->getAtIndices(mCurrentIndices);
      }
    }

    // prefix increment
    SelfType& operator++()
    {
      mCurrentIndices = mParent->nextIndices(mCurrentIndices);
      if (mCurrentIndices != mParent->endIndices()) {
        mElement = mParent->getAtIndices(mCurrentIndices);
      } else {
        mElement = ElementType{};
      }
      return *this;
    }
    // postfix increment
    SelfType operator++(int /*unused*/)
    {
      SelfType copy(*this);
      operator++();
      return copy;
    }

    // return reference
    reference operator*() const
    {
      return mElement;
    }

    bool operator==(const SelfType& rh) const
    {
      return mCurrentIndices == rh.mCurrentIndices;
    }

    auto operator<=>(const SelfType& rh) const
    {
      return mCurrentIndices <=> rh.mCurrentIndices;
    }

    // return pointer to parent instance
    [[nodiscard]] ParentType const* parent() const
    {
      return mParent;
    }

    // return current position (headerIdx serves as the slot index for slot-level iteration)
    [[nodiscard]] size_t position() const
    {
      return mCurrentIndices.headerIdx;
    }

    // return an iterable range over all parts in the current slot
    // only available for slot-level iterators whose parent has parts(size_t)
    [[nodiscard]] auto parts() const
      requires requires(ParentType const* p, size_t i) { p->parts(i); }
    {
      return mParent->parts(mCurrentIndices.headerIdx);
    }

   private:
    ParentType const* mParent;
    DataRefIndices mCurrentIndices;
    ElementType mElement;
  };

  /// A range over the parts of a single slot, supporting range-based for.
  struct PartRange {
    InputSpan const* span;
    size_t slot;

    [[nodiscard]] DataRefIndices initialIndices() const { return {0, 1}; }
    [[nodiscard]] DataRefIndices endIndices() const { return {size_t(-1), size_t(-1)}; }
    [[nodiscard]] DataRef getAtIndices(DataRefIndices idx) const { return span->getAtIndices(slot, idx); }
    [[nodiscard]] DataRefIndices nextIndices(DataRefIndices idx) const { return span->nextIndices(slot, idx); }
    [[nodiscard]] size_t size() const { return span->getNofParts(slot); }

    [[nodiscard]] Iterator<PartRange, const DataRef> begin() const { return {this, size() == 0}; }
    [[nodiscard]] Iterator<PartRange, const DataRef> end() const { return {this, true}; }
  };

  /// Return an iterable range over all parts in slot @a i.
  [[nodiscard]] PartRange parts(size_t i) const { return {this, i}; }

  using const_iterator = Iterator<InputSpan, const DataRef>;
  using iterator = const_iterator;

  // supporting read-only access and returning const_iterator
  [[nodiscard]] const_iterator begin() const
  {
    return {this, false};
  }

  [[nodiscard]] const_iterator end() const
  {
    return {this, true};
  }

 private:
  std::function<size_t(size_t)> mNofPartsGetter;
  std::function<int(size_t)> mRefCountGetter;
  std::function<DataRef(size_t, DataRefIndices)> mIndicesGetter;
  std::function<DataRefIndices(size_t, DataRefIndices)> mNextIndicesGetter;
  size_t mSize;
};

} // namespace o2::framework

#endif // FRAMEWORK_INPUTSSPAN_H
