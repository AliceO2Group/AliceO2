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

extern template class std::function<o2::framework::DataRef(size_t)>;
extern template class std::function<o2::framework::DataRef(size_t, size_t)>;

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

  /// @a getter is the mapping between an element of the span referred by
  /// index and the buffer associated.
  /// @a size is the number of elements in the span.
  InputSpan(std::function<DataRef(size_t)> getter, size_t size);

  /// @a getter is the mapping between an element of the span referred by
  /// index and the buffer associated.
  /// @a size is the number of elements in the span.
  InputSpan(std::function<DataRef(size_t, size_t)> getter, size_t size);

  /// @a getter is the mapping between an element of the span referred by
  /// index and the buffer associated.
  /// @nofPartsGetter is the getter for the number of parts associated with an index
  /// @a size is the number of elements in the span.
  InputSpan(std::function<DataRef(size_t, size_t)> getter, std::function<size_t(size_t)> nofPartsGetter, size_t size);

  /// @a i-th element of the InputSpan
  [[nodiscard]] DataRef get(size_t i, size_t partidx = 0) const
  {
    return mGetter(i, partidx);
  }

  /// @a number of parts in the i-th element of the InputSpan
  [[nodiscard]] size_t getNofParts(size_t i) const
  {
    if (i >= mSize) {
      return 0;
    }
    if (!mNofPartsGetter) {
      return 1;
    }
    return mNofPartsGetter(i);
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

  /// an iterator class working on position within the a parent class
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

    Iterator(ParentType const* parent, size_t position = 0, size_t size = 0)
      : mPosition(position), mSize(size > position ? size : position), mParent(parent), mElement{}
    {
      if (mPosition < mSize) {
        mElement = mParent->get(mPosition);
      }
    }

    ~Iterator() = default;

    // prefix increment
    SelfType& operator++()
    {
      if (mPosition < mSize && ++mPosition < mSize) {
        mElement = mParent->get(mPosition);
      } else {
        // reset the element to the default value of the type
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

    // comparison
    bool operator==(const SelfType& rh) const
    {
      return mPosition == rh.mPosition;
    }

    // comparison
    bool operator!=(const SelfType& rh) const
    {
      return mPosition != rh.mPosition;
    }

    // return pointer to parent instance
    [[nodiscard]] ParentType const* parent() const
    {
      return mParent;
    }

    // return current position
    [[nodiscard]] size_t position() const
    {
      return mPosition;
    }

   private:
    size_t mPosition;
    size_t mSize;
    ParentType const* mParent;
    ElementType mElement;
  };

  /// @class InputSpanIterator
  /// An iterator over the input slots
  /// It supports an iterator interface to access the parts in the slot
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

    InputSpanIterator(InputSpan const* parent, size_t position = 0, size_t size = 0)
      : BaseType(parent, position, size)
    {
    }

    /// Get element at {slotindex, partindex}
    [[nodiscard]] ElementType get(size_t pos) const
    {
      return this->parent()->get(this->position(), pos);
    }

    /// Check if slot is valid, index of part is not used
    [[nodiscard]] bool isValid(size_t = 0) const
    {
      if (this->position() < this->parent()->size()) {
        return this->parent()->isValid(this->position());
      }
      return false;
    }

    /// Get number of parts in input slot
    [[nodiscard]] size_t size() const
    {
      return this->parent()->getNofParts(this->position());
    }

    // iterator for the part access
    [[nodiscard]] const_iterator begin() const
    {
      return const_iterator(this, 0, size());
    }

    [[nodiscard]] const_iterator end() const
    {
      return const_iterator(this, size());
    }
  };

  using iterator = InputSpanIterator<DataRef>;
  using const_iterator = InputSpanIterator<const DataRef>;

  // supporting read-only access and returning const_iterator
  [[nodiscard]] const_iterator begin() const
  {
    return {this, 0, size()};
  }

  // supporting read-only access and returning const_iterator
  [[nodiscard]] const_iterator end() const
  {
    return {this, size()};
  }

 private:
  std::function<DataRef(size_t, size_t)> mGetter;
  std::function<size_t(size_t)> mNofPartsGetter;
  size_t mSize;
};

} // namespace o2::framework

#endif // FRAMEWORK_INPUTSSPAN_H
