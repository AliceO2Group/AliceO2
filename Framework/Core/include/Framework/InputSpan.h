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

   private:
    ParentType const* mParent;
    DataRefIndices mCurrentIndices;
    ElementType mElement;
  };

  /// @class InputSpanIterator
  /// An iterator over the input slots.
  /// It supports an iterator interface to access the parts in the slot.
  template <typename T>
  class InputSpanIterator : public Iterator<InputSpan, T>
  {
   public:
    using SelfType = InputSpanIterator;
    using BaseType = Iterator<InputSpan, T>;
    using value_type = typename BaseType::value_type;
    using reference = typename BaseType::reference;
    using pointer = typename BaseType::pointer;
    using ElementType = typename std::remove_const<value_type>::type;
    using iterator = Iterator<SelfType, T>;
    using const_iterator = Iterator<SelfType, const T>;

    InputSpanIterator(InputSpan const* parent, bool isEnd = false)
      : BaseType(parent, isEnd)
    {
    }

    /// Initial indices for part-level iteration: first part starts at {headerIdx=0, payloadIdx=1}.
    [[nodiscard]] DataRefIndices initialIndices() const { return {0, 1}; }
    /// Sentinel used by nextIndicesGetter to signal end-of-slot.
    [[nodiscard]] DataRefIndices endIndices() const { return {size_t(-1), size_t(-1)}; }

    /// Get element at the given raw message indices in O(1).
    [[nodiscard]] ElementType getAtIndices(DataRefIndices indices) const
    {
      return this->parent()->getAtIndices(this->position(), indices);
    }

    /// Advance @a current to the next part's indices in O(1).
    [[nodiscard]] DataRefIndices nextIndices(DataRefIndices current) const
    {
      return this->parent()->nextIndices(this->position(), current);
    }

    /// Get number of parts in input slot
    [[nodiscard]] size_t size() const
    {
      return this->parent()->getNofParts(this->position());
    }

    [[nodiscard]] const_iterator begin() const
    {
      return const_iterator(this, size() == 0);
    }

    [[nodiscard]] const_iterator end() const
    {
      return const_iterator(this, true);
    }
  };

  using iterator = InputSpanIterator<DataRef>;
  using const_iterator = InputSpanIterator<const DataRef>;

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
