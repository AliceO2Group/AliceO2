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
         typename ElementType,
         typename GroupT = int // later extension to groups of elements
         >
class PageParser {
public:
  using PageHeaderType = PageHeaderT;
  using BufferType = unsigned char;
  using value_type = ElementType;
  using GroupType = GroupT;
  using GetNElements = std::function<size_t(const GroupType&)>;
  static const size_t page_size = PageSize;

  PageParser() = delete;
  PageParser(const BufferType* buffer, size_t size,
             GetNElements getNElementsFct = [] (const GroupType&) {return 0;}
             )
    : mBuffer(buffer)
    , mSize(size)
    , mGetNElementsFct(getNElementsFct)
    , mNPages(size>0?(size/page_size)+1:0)
  {
  }
  ~PageParser() = default;

  using IteratorBase = std::iterator<std::forward_iterator_tag, value_type>;

  class Iterator : public IteratorBase {
  public:
    using ParentType = PageParser;
    using SelfType = Iterator;
    using value_type = typename IteratorBase::value_type;
    using reference = typename IteratorBase::reference;
    using pointer = typename IteratorBase::pointer;

    Iterator() = delete;

    Iterator(const ParentType & parent, size_t position = 0)
      : mParent(parent)
    {
      mPosition = mParent.getElement(position, mElement);
    }
    ~Iterator() = default;

    // prefix increment
    SelfType& operator++() {
      mPosition = mParent.getElement(mPosition, mElement);
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
    int mPosition;
    const ParentType& mParent;
    value_type mElement;
  };

  /// retrieve an object at position
  size_t getElement(size_t position, value_type& element) const {
    // check if we are at the end
    if (position == mSize) return position;

    // check if there is space for one element
    if (position + sizeof(value_type) > mSize) {
      // format error, probably throw exception
      return mSize;
    }
    auto copySize = sizeof(value_type);
    auto target = reinterpret_cast<BufferType*>(&element);
    if ((position % page_size) == 0) {
      // skip the page header at beginning of page
      position += sizeof(PageHeaderType);
    }
    if ((position % page_size) + copySize > page_size) {
      // object is split at the page boundary, copy the part
      // in the current page first
      copySize = ((position % page_size) + copySize) - page_size;
    }
    if (copySize > 0) {
      memcpy(target, mBuffer + position, copySize);
      position += copySize;
      target += copySize;
    }
    copySize = sizeof(value_type) - copySize;
    if (copySize > 0) {
      // skip page header at beginning of new page and copy
      // remaining part of the element
      position += sizeof(PageHeaderType);
      memcpy(target, mBuffer + position, copySize);
      position += copySize;
    }
    return position;
  }

  Iterator begin() const {
    return Iterator(*this, 0);
  }

  Iterator end() const {
    return Iterator(*this, mSize);
  }

private:
  const BufferType* mBuffer = nullptr;
  size_t mSize = 0;
  GetNElements mGetNElementsFct;
  size_t mNPages = 0;
};

}
}

#endif
