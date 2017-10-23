// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALGORITHM_PAGEPARSER_H
#define ALGORITHM_PAGEPARSER_H

/// @file   PageParser.h
/// @author Matthias Richter
/// @since  2017-09-27
/// @brief  Parser for a set of data objects in consecutive memory pages.

#include <functional>
#include <vector>
#include <cassert>
#include <type_traits>
#include <stdexcept>

namespace o2 {

namespace algorithm {

/**
 * @class PageParser
 * Parser for a set of data objects in consecutive memory pages.
 *
 * All memory pages have a fixed size and start with a page header.
 * Depending on the page size and size of the data object, some
 * objects can be split at the page boundary and have the page header
 * embedded.
 *
 * The class iterator can be used to iterate over the data objects
 * transparently.
 *
 * Usage:
 *   RawParser<PageHeaderType, N, ElementType> RawParser;
 *   RawParser parser(ptr, size);
 *   for (auto element : parser) {
 *     // do something with element
 *   }
 */
template<typename PageHeaderT,
         size_t PageSize,
         typename ElementT,
         typename GroupT = int // later extension to groups of elements
         >
class PageParser {
public:
  using PageHeaderType = PageHeaderT;
  using BufferType = unsigned char;
  using value_type = ElementT;
  using GroupType = GroupT;
  using GetNElements = std::function<size_t(const GroupType&)>;
  static const size_t page_size = PageSize;

  // at the moment an object can only be split among two pages
  static_assert(PageSize >= sizeof(PageHeaderType) + sizeof(value_type),
                "Page Header and at least one element have to fit into page");

  // switches for the copy method, used to skip ill-formed expressions
  using TargetInPageBuffer = std::true_type;
  using SourceInPageBuffer = std::false_type;

  PageParser() = delete;
  template<typename T>
  PageParser(T* buffer, size_t size,
             GetNElements getNElementsFct = [] (const GroupType&) {return 0;}
             )
    : mBuffer(nullptr)
    , mBufferIsConst(std::is_const<T>::value)
    , mSize(size)
    , mGetNElementsFct(getNElementsFct)
    , mNPages(size>0?((size-1)/page_size)+1:0)
  {
    static_assert(sizeof(T) == sizeof(BufferType),
                  "buffer required to be byte-type");

    // the buffer pointer is stored non-const, a runtime check ensures
    // that iterator write works only for non-const buffers
    mBuffer = const_cast<BufferType*>(buffer);
  }
  ~PageParser() = default;

  template<typename T>
  using IteratorBase = std::iterator<std::forward_iterator_tag, T>;

  template<typename T>
  class Iterator : public IteratorBase<T> {
  public:
    using ParentType = PageParser;
    using SelfType = Iterator;
    using value_type = typename IteratorBase<T>::value_type;
    using reference = typename IteratorBase<T>::reference;
    using pointer = typename IteratorBase<T>::pointer;
    using ElementType = typename std::remove_const<value_type>::type;


    Iterator() = delete;

    Iterator(ParentType const * parent, size_t position = 0)
      : mParent(parent)
    {
      mPosition = position;
      mNextPosition = mParent->getElement(mPosition, mElement);
      backup();
    }
    ~Iterator()
    {
      sync();
    }

    // prefix increment
    SelfType& operator++() {
      sync();
      mPosition = mNextPosition;
      mNextPosition = mParent->getElement(mPosition, mElement);
      backup();
      return *this;
    }
    // postfix increment
    SelfType operator++(int /*unused*/) {
      SelfType copy(*this); operator++(); return copy;
    }
    // return reference
    reference operator*() {
      return mElement;
    }
    // comparison
    bool operator==(const SelfType& rh) {
      return mPosition == rh.mPosition;
    }
    // comparison
    bool operator!=(const SelfType& rh) {
      return mPosition != rh.mPosition;
    }

  private:
    // sync method for non-const iterator
    template< typename U = void >
    typename std::enable_if< !std::is_const<value_type>::value, U >::type sync() {
      if (std::memcmp(&mElement, &mBackup, sizeof(value_type)) != 0) {
        // mElement is changed, sync to buffer
        mParent->setElement(mPosition, mElement);
      }
    }

    // overload for const_iterator, empty function body
    template< typename U = void >
    typename std::enable_if< std::is_const<value_type>::value, U >::type sync() {}

    // backup for non-const iterator
    template< typename U = void >
    typename std::enable_if< !std::is_const<value_type>::value, U >::type backup() {
      mBackup = mElement;
    }

    // overload for const_iterator, empty function body
    template< typename U = void >
    typename std::enable_if< std::is_const<value_type>::value, U >::type backup() {}

    int mPosition;
    int mNextPosition;
    ParentType const * mParent;
    ElementType mElement;
    ElementType mBackup;
  };

  /// set an object at position
  size_t setElement(size_t position, const value_type& element) const {
    // check if we are at the end
    if (position >= mSize) {
      assert(position == mSize);
      return mSize;
    }

    // check if there is space for one element
    if (position + sizeof(value_type) > mSize) {
      // format error, probably throw exception
      return mSize;
    }

    auto source = reinterpret_cast<const BufferType*>(&element);
    auto target = mBuffer + position;
    return position + copy<TargetInPageBuffer>(source, target, page_size - (position % page_size));
  }

  /// retrieve an object at position
  size_t getElement(size_t position, value_type& element) const {
    // check if we are at the end
    if (position >= mSize) {
      assert(position == mSize);
      return mSize;
    }

    // check if there is space for one element
    if (position + sizeof(value_type) > mSize) {
      // format error, probably throw exception
      return mSize;
    }

    auto source = mBuffer + position;
    auto target = reinterpret_cast<BufferType*>(&element);
    return position + copy<SourceInPageBuffer>(source, target, page_size - (position % page_size));
  }

  // copy data, depending on compile time switch, either source or target
  // pointer are treated as pointer in the raw page, i.e. can be additionally
  // incremented by the page header
  template<typename SwitchT>
  size_t copy(const BufferType* source, BufferType* target, size_t pageCapacity) const
  {
    size_t position = 0;
    auto copySize = sizeof(value_type);
    // choose which of the pointers needs additional PageHeader offsets
    auto pageOffsetTarget = SwitchT::value? &target : const_cast<BufferType**>(&source);
    if (pageCapacity == page_size) {
      // skip the page header at beginning of page
      position += sizeof(PageHeaderType);
      pageCapacity -= sizeof(PageHeaderType);
      *pageOffsetTarget += sizeof(PageHeaderType);
    }
    if (copySize > pageCapacity) {
      // object is split at the page boundary, copy the part
      // in the current page first
      copySize = pageCapacity;
    }
    if (copySize > 0) {
      memcpy(target, source, copySize);
      position += copySize;
      source += copySize;
      target += copySize;
    }
    copySize = sizeof(value_type) - copySize;
    if (copySize > 0) {
      // skip page header at beginning of new page and copy
      // remaining part of the element
      position += sizeof(PageHeaderType);
      *pageOffsetTarget += sizeof(PageHeaderType);
      memcpy(target, source, copySize);
      position += copySize;
    }
    return position;
  }

  using iterator = Iterator<value_type>;
  using const_iterator = Iterator<const value_type>;

  const_iterator begin() const {
    return const_iterator(this, 0);
  }

  const_iterator end() const {
    return const_iterator(this, mSize);
  }

  iterator begin() {
    if (mBufferIsConst) {
      // did not find a way to do this at compile time in the constructor,
      // probably one needs to make the buffer type a template parameter
      // to the class
      throw std::runtime_error("the underlying buffer is not writeable");
    }
    return iterator(this, 0);
  }

  iterator end() {
    return iterator(this, mSize);
  }

private:
  BufferType* mBuffer = nullptr;
  bool mBufferIsConst = false;
  size_t mSize = 0;
  GetNElements mGetNElementsFct;
  size_t mNPages = 0;
};

}
}

#endif
