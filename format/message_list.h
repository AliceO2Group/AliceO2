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
#include "memory-format.h"

namespace AliceO2 {
namespace Format {

// Ideally it does not matter for the implementation of the container class
// whether the type is the message container or the pointer to the payload
template<class MsgT, class HdrT>
class messageList {
 public:
  typedef MsgT message_type;
  typedef HdrT header_type;

  messageList() {}
  messageList(const messageList& other); // not yet implemented
  messageList& operator=(const messageList& other); // not yet implemented
  ~messageList() {}

  /** add data block to list */
  int add(MsgT& headerMsg, MsgT& payloadMsg) {
    // conversion relies on the conversion operator for complex types
    const uint8_t* headerData = headerMsg;

    const HdrT* srcHeader = reinterpret_cast<const HdrT*>(headerData);
    // TODO: consistency check
    mDataArray.push_back(messagePair(*srcHeader, payloadMsg));
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

  class iterator {
   public:
    typedef iterator self_type;
    typedef MsgT value_type;
  
    iterator(const pairIt_t& dataIterator) : mDataIterator(dataIterator) { }
    // prefix increment
    self_type operator++() { ++mDataIterator; return *this; }
    // postfix increment
    self_type operator++(int unused) {self_type copy(*this); ++*this; return copy;}
    // TODO: given the fact that the data which is hold is a pointer, it always needs to be
    // valid for dereference
    MsgT& operator*() { return *((*mDataIterator).mPayload);}
    MsgT* operator->() { return (*mDataIterator).mPayload;}

    HdrT& getHdr() const {return (*mDataIterator).mHeader;}

    bool operator==(const self_type& other) { return mDataIterator == other.mDataIterator; }
    bool operator!=(const self_type& other) { return mDataIterator != other.mDataIterator; }

    /** TODO: comparison object to be used
    bool operator==(const PayloadMetaData_t& md) { return (*mDataIterator).mHeader == md; }
    bool operator!=(const PayloadMetaData_t& md) { return (*mDataIterator).mHeader != md; }
    */

    /** conversion operator to PayloadMetaData_t struct */
    operator HdrT() {return (*mDataIterator).mHeader;}
    /** return size of payload */
    size_t size() {return (*mDataIterator).mHeader.mPayloadSize;}

  private:
    pairIt_t mDataIterator;
  };

  /** to be defined
  class const_iterator
  {
  };
  */

  iterator begin() {
    pairIt_t it = mDataArray.begin();
    return iterator(it);
  }

  iterator end() {
    pairIt_t it = mDataArray.end();
    return iterator(it);
  }

 private:
  std::vector<messagePair> mDataArray;
};

}; // namespace Format
}; // namespace AliceO2
#endif
