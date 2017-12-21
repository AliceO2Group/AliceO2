// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_INPUTRECORD_H
#define FRAMEWORK_INPUTRECORD_H

#include "Framework/DataRef.h"
#include "Framework/DataRefUtils.h"
#include "Framework/InputRoute.h"

#include <fairmq/FairMQMessage.h>
#include <Framework/TMessageSerializer.h>

#include <TClass.h>

#include <iterator>
#include <string>
#include <vector>
#include <cstring>
#include <cassert>
#include <exception>
#include <memory>
#include <type_traits>

namespace o2 {
namespace framework {

struct InputSpec;

/// This class holds the inputs which  are being processed by the system while
/// they are  being processed.  The user can  get an instance  for it  via the
/// ProcessingContext and can use it to retrieve the inputs, either by name or
/// by index.  A few utility  methods are  provided to automatically  cast the
/// inputs to  known types. The user is also allowed to  override the `get`
/// template and provide his own serialization mechanism.
class InputRecord {
public:
  InputRecord(std::vector<InputRoute> const &inputs,
              std::vector<std::unique_ptr<FairMQMessage>> const &cache);

  int getPos(const char *name) const;
  int getPos(const std::string &name) const;

  DataRef getByPos(int pos) const {
    if (pos*2 >= mCache.size() || pos < 0) {
      throw std::runtime_error("Unknown argument requested at position " + std::to_string(pos));
    }
    assert(pos >= 0);
    return DataRef{&mInputsSchema[pos].matcher,
                   static_cast<char const*>(mCache[pos*2]->GetData()),
                   static_cast<char const*>(mCache[pos*2+1]->GetData())};
  }

  // Generic function to automatically cast the contents of 
  // a payload bound by @a binding to a known type. This will
  // not be used if the type is a TObject as extra deserialization
  // needs to happen.
  template <typename T>
  typename std::enable_if<std::is_pod<T>::value && std::is_same<T, DataRef>::value == false, T>::type const&
  get(char const *binding) const {
    return *reinterpret_cast<T const *>(get<DataRef>(binding).payload);
  }

  // If we ask for a char const *, we simply point to the payload. Notice this
  // is meant for C-style strings. If you want to actually get hold of the buffer,
  // use get<DataRef> (or simply get) as that will give you the size as well.
  // FIXME: check that the string is null terminated.
  template <typename T>
  typename std::enable_if<std::is_same<T, char const *>::value, T>::type
  get(char const *binding) const {
    return reinterpret_cast<char const *>(get<DataRef>(binding).payload);
  }

  // If we ask for a string, we need to duplicate it because we do not want
  // the buffer to be deleted when it goes out of scope.
  // FIXME: check that the string is null terminated.
  template <typename T>
  typename std::enable_if<std::is_same<T, std::string>::value, T>::type
  get(char const *binding) const {
    return std::string(get<DataRef>(binding).payload);
  }

  // DataRef is special. Since there is no point in storing one in a payload,
  // what it actually does is to return the DataRef used to hold the 
  // (header, payload) pair.
  template <typename T = DataRef>
  typename std::enable_if<std::is_same<T, DataRef>::value, T>::type
  get(const char *binding) const {
    try {
      return getByPos(getPos(binding));
    } catch(...) {
      throw std::runtime_error("Unknown argument requested" + std::string(binding));
    }
  }

  // DataRef is special. Since there is no point in storing one in a payload,
  // what it actually does is to return the DataRef used to hold the 
  // (header, payload) pair.
  template <class T = DataRef>
  typename std::enable_if<std::is_same<T, DataRef>::value, T>::type
  get(std::string const &binding) const {
    return getByPos(getPos(binding));
  }

  // Notice that this will return a copy of the actual contents of
  // the buffer, because the buffer is actually serialised, for this
  // reason we return a unique_ptr<T>.
  // FIXME: does it make more sense to keep ownership of all the deserialised 
  // objects in a single place so that we can avoid duplicate deserializations?
  template <class T>
  typename std::unique_ptr<typename std::enable_if<std::is_base_of<TObject, T>::value == true, T>::type>
  get(char const *binding) const {
    using DataHeader = o2::header::DataHeader;

    auto ref = this->get(binding);
    return std::move(DataRefUtils::as<T>(ref));
  }

  size_t size() const {
    return mCache.size()/2;
  }

  template<typename T>
  using IteratorBase = std::iterator<std::forward_iterator_tag, T>;

  template<typename ParentT, typename T>
  class Iterator : public IteratorBase<T> {
  public:
    using ParentType = ParentT;
    using SelfType = Iterator;
    using value_type = typename IteratorBase<T>::value_type;
    using reference = typename IteratorBase<T>::reference;
    using pointer = typename IteratorBase<T>::pointer;
    using ElementType = typename std::remove_const<value_type>::type;

    Iterator() = delete;

  Iterator(ParentType const * parent, size_t position = 0, size_t size = 0)
      : mParent(parent)
      , mPosition(position)
      , mSize(size > position? size : position)
      , mElement{nullptr, nullptr, nullptr}
    {
      if (mPosition < mSize) {
        mElement = mParent->getByPos(mPosition);
      }
    }

    ~Iterator() = default;

    // prefix increment
    SelfType& operator++() {
      if (mPosition < mSize && ++mPosition < mSize) {
        mElement = mParent->getByPos(mPosition);
      }
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
    size_t mPosition;
    size_t mSize;
    ParentType const * mParent;
    ElementType mElement;
  };

  using iterator = Iterator<InputRecord, DataRef>;
  using const_iterator = Iterator<InputRecord, const DataRef>;

  const_iterator begin() const {
    return const_iterator(this, 0, size());
  }

  const_iterator end() const {
    return const_iterator(this, size());
  }

private:
  std::vector<InputRoute> const &mInputsSchema;
  std::vector<std::unique_ptr<FairMQMessage>> const &mCache;
};

} // framework
} // o2

#endif // FRAMEWORK_INPUTREGISTRY_H
