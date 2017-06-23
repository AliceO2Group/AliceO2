// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_COLLECTION_H
#define FRAMEWORK_COLLECTION_H

#include <cstddef>
#include <cassert>
#include <type_traits>

// FIXME: not sure if something like this already exists in C++

/// Given a fixed char * array, give an iterator like way to access it.

namespace o2 {
namespace framework {

template <typename T>
class CollectionIt {
public:
  CollectionIt(T *ptr)
  :mPtr{ptr}{
  }

  T &operator*(void) const {
    return *mPtr;
  }

  T &operator->(void) {
    return *mPtr;
  }

  CollectionIt<T> operator++(int /*dummy*/) {
    CollectionIt<T> tmp(mPtr);
    mPtr++;
    assert((char*) mPtr - (char*)tmp == sizeof(T));
    return tmp;
  }

  CollectionIt<T> operator++(void) {
    mPtr++;
    return *this;
  }

  CollectionIt<T> operator--(int /*dummy*/) {
    CollectionIt<T> tmp(mPtr);
    mPtr--;
    return tmp;
  }

  CollectionIt<T> operator--(void) {
    mPtr--;
    return *this;
  }

  CollectionIt<T>& operator+=(int delta)
  {
    mPtr += delta;
    return *this;
  }

  CollectionIt<T>& operator-=(int delta)
  {
    mPtr -= delta;
    return *this;
  }

  CollectionIt<T>& operator=(int value)
  {
    mPtr = value;
    return *this;
  }

  bool operator==(const CollectionIt<T>& other)
  {
    return mPtr == other.mPtr;
  }

  bool operator!=(const CollectionIt<T>& other)
  {
    return mPtr != other.mPtr;
  }

  CollectionIt<T> operator+(int value)
  {
    return CollectionIt<T>(mPtr+value);
  }
private:
  T *mPtr;
};

template <typename T>
class Collection {
public:
  using iterator = CollectionIt<T>;

  Collection(void *data, size_t size)
  : mData{reinterpret_cast<T*>(data)},
    mSize{size}
  {
    static_assert(std::is_pod<T>::value == true, "Collection only works with PoD");
  }

  // Collection is non-owning, since the data is assumed to be created
  // by either the transport or to be adopted.
  ~Collection() {}

  size_t size()
  {
    return mSize;
  }

  T &at(size_t pos)
  {
    return *(mData + pos);
  }

  iterator begin()
  {
    return iterator(mData);
  }

  iterator end()
  {
    return iterator(mData+mSize);
  }

  iterator cbegin() const
  {
    return iterator(mData);
  }

  iterator cend() const
  {
    return iterator(mData+mSize);
  }

private:
  T *mData;
  size_t mSize;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_COLLECTION_H
