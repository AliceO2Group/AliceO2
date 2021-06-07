// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_INPUTSPAN_H
#define FRAMEWORK_INPUTSPAN_H

#include "Framework/DataRef.h"
#include <functional>

extern template class std::function<o2::framework::DataRef(size_t)>;
extern template class std::function<o2::framework::DataRef(size_t, size_t)>;

namespace o2::framework
{

/// Mapping helper between the store of all inputs being processed and the
/// actual inputs to be processed in a given go.
/// In general this will use an helper which returns
/// `FairMQMessages->GetData()` from the Message cache, but in principle
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
  DataRef get(size_t i, size_t partidx = 0) const
  {
    return mGetter(i, partidx);
  }

  /// @a number of parts in the i-th element of the InputSpan
  size_t getNofParts(size_t i) const
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
  size_t size() const
  {
    return mSize;
  }

  const char* header(size_t i) const
  {
    return get(i).header;
  }

  const char* payload(size_t i) const
  {
    return get(i).payload;
  }

  template <typename T>
  using IteratorBase = std::iterator<std::forward_iterator_tag, T>;

  /// an iterator class working on position within the a parent class
  template <typename ParentT, typename T>
  class Iterator : public IteratorBase<T>
  {
   public:
    using ParentType = ParentT;
    using SelfType = Iterator;
    using value_type = typename IteratorBase<T>::value_type;
    using reference = typename IteratorBase<T>::reference;
    using pointer = typename IteratorBase<T>::pointer;
    using ElementType = typename std::remove_const<value_type>::type;

    Iterator() = delete;

    Iterator(ParentType const* parent, size_t position = 0, size_t size = 0)
      : mParent(parent), mPosition(position), mSize(size > position ? size : position), mElement{}
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
    ParentType const* parent() const
    {
      return mParent;
    }

    // return current position
    size_t position() const
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
    ElementType get(size_t pos) const
    {
      return this->parent()->get(this->position(), pos);
    }

    /// Check if slot is valid, index of part is not used
    bool isValid(size_t = 0) const
    {
      if (this->position() < this->parent()->size()) {
        return this->parent()->isValid(this->position());
      }
      return false;
    }

    /// Get number of parts in input slot
    size_t size() const
    {
      return this->parent()->getNofParts(this->position());
    }

    // iterator for the part access
    const_iterator begin() const
    {
      return const_iterator(this, 0, size());
    }

    const_iterator end() const
    {
      return const_iterator(this, size());
    }
  };

  using iterator = InputSpanIterator<DataRef>;
  using const_iterator = InputSpanIterator<const DataRef>;

  // supporting read-only access and returning const_iterator
  const_iterator begin() const
  {
    return const_iterator(this, 0, size());
  }

  // supporting read-only access and returning const_iterator
  const_iterator end() const
  {
    return const_iterator(this, size());
  }

 private:
  std::function<DataRef(size_t, size_t)> mGetter;
  std::function<size_t(size_t)> mNofPartsGetter;
  size_t mSize;
};

} // namespace o2::framework

#endif // FRAMEWORK_INPUTSSPAN_H
