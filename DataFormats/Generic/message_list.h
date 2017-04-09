//-*- Mode: C++ -*-

#ifndef MESSAGELIST_H
#define MESSAGELIST_H
//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or	     *
//* (at your option) any later version.					     *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   message_list.h
//  @author Matthias Richter
//  @since  2016-02-11
//  @brief  Container class for messages in the O2 framework

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring> // memset
#include <functional> // std::function

namespace o2 {
namespace Format {

// Ideally it does not matter for the implementation of the container class
// whether the type is the message container or the pointer to the payload
template<class MsgT, class HdrT>
class messageList {
 public:
  typedef MsgT message_type;
  typedef HdrT header_type;
  /// comparison metric for selection of elements
  /// an external function can be defined by the caller of begin() to
  /// apply a selection of elements
  typedef std::function<bool(const HdrT& hdr)> HdrComparison;

  messageList() = default;
  messageList(const messageList& other); // not yet implemented
  messageList& operator=(const messageList& other); // not yet implemented
  ~messageList() = default;

  /// add data block to list
  /// both header and payload message parts are required to add an entry
  /// the actual header of type HdrT is extracted from the header
  /// message part.
  int add(MsgT& headerMsg, MsgT& payloadMsg) {
    // conversion relies on the conversion operator for complex types
    const uint8_t* headerData = headerMsg;

    const HdrT* srcHeader = reinterpret_cast<const HdrT*>(headerData);
    // TODO: consistency check
    mDataArray.push_back(messagePair(*srcHeader, payloadMsg));

    return mDataArray.size();
  }
  /** number of data blocks in the list */
  size_t size() {return mDataArray.size();}
  /** clear the list */
  void clear() {mDataArray.clear();}
  /** check if list is empty */
  bool empty() {mDataArray.empty();}

  /**
   * messagePair describes the two sequential message parts for header and payload
   * respectively.
   *
   * TODO: decide whether to use pointer to message or pointer to payload in the
   * message. Whith the template approach and appropriate conversion operators in the
   * message class possibly both ways can be served at the same time
   */
  struct messagePair {
    HdrT  mHeader;
    MsgT* mPayload;

    messagePair(MsgT& payload) : mHeader(), mPayload(&payload) {
      memset(&mHeader, 0, sizeof(HdrT));
    }

    messagePair(const HdrT& header, MsgT& payload) : mHeader(), mPayload(&payload) {
      memcpy(&mHeader, &header, sizeof(HdrT));
    }
  };
  typedef typename std::vector<messagePair>::iterator pairIt_t;
  // TODO: operators inside a class can only have one parameter
  // check whether to create a functor class
  //bool operator==(const messageList::pairIt_t& first, const messageList::pairIt_t& second) {
  //  return (first->mHeader == second->mHeader) && (first->mPayload == second->mPayload);
  //}
  //
  //bool operator!=(const messageList::pairIt_t& first, const messageList::pairIt_t& second) {
  //  return (first->mHeader != second->mHeader) || (first->mPayload != second->mPayload);
  //}

  /**
   * @class iterator
   * Implementation of navigation through the list and access to elements.
   *
   * An optional comparison metric @ref HdrComparison can be used to provide
   * a selection of elements.
   */
  class iterator {
   public:
    typedef iterator self_type;
    typedef MsgT value_type;

    iterator(const pairIt_t& dataIterator, const pairIt_t& iteratorRange,
	     const HdrComparison hdrsel = HdrComparison())
      : mDataIterator(dataIterator)
      , mEnd(iteratorRange)
      , mHdrSelection(hdrsel)
    { }
    iterator(const pairIt_t& dataIterator)
      : mDataIterator(dataIterator)
      , mEnd(dataIterator)
      , mHdrSelection(HdrComparison())
    { }
    // prefix increment
    self_type& operator++() {
      while (++mDataIterator != mEnd) {
	// operator bool() of std::function is used to determine whether
	// a selector is set or not, the default is not callable.
	// if the std::function container has an assigned target, this is
	// called with the header as parameter
	if (!mHdrSelection || mHdrSelection(mDataIterator->mHeader)) break;
      }
      return *this;
    }
    // postfix increment
    self_type operator++(int unused) {self_type copy(*this); ++*this; return copy;}
    // TODO: given the fact that the data which is hold is a pointer, it always needs to be
    // valid for dereference
    MsgT& operator*() { return *((*mDataIterator).mPayload);}
    MsgT* operator->() { return (*mDataIterator).mPayload;}

    /** return header at iterator position */
    HdrT& getHdr() const {return (*mDataIterator).mHeader;}

    bool operator==(const self_type& other) { return mDataIterator == other.mDataIterator; }
    bool operator!=(const self_type& other) { return mDataIterator != other.mDataIterator; }

    /** conversion operator to PayloadMetaData_t struct */
    operator HdrT() {return (*mDataIterator).mHeader;}
    /** return size of payload */
    size_t size() {return (*mDataIterator).mHeader.mPayloadSize;}

  private:
    pairIt_t mDataIterator;
    pairIt_t mEnd;
    HdrComparison mHdrSelection;
  };

  /** to be defined
  class const_iterator
  {
  };
  */

  iterator begin(const HdrComparison hdrsel = HdrComparison()) {
    iterator ret(mDataArray.begin(), mDataArray.end(), hdrsel);
    // if the std::function container has an assigned target, this is
    // is used used to check whether the iterator matches the selection
    // further checks are in the increment operator.
    // the iterator class implements a type cast operator which allows
    // to use it directly in the HdrComparison
    if (hdrsel && !hdrsel(ret)) ++ret;
    return ret;
  }

  iterator end() {
    return iterator(mDataArray.end());
  }

 private:
  std::vector<messagePair> mDataArray;
};

}; // namespace Format
}; // namespace AliceO2
#endif
