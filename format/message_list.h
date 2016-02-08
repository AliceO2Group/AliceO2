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

//  @file   messageList.h
//  @author Matthias Richter
//  @since  2016-02-11
//  @brief  Container class for messages in the O2 framework

#include "memory-format.h"

namespace AliceO2 {
namespace Format {

// Ideally it does not matter for the implementation of the container class
// whether the type is the message container or the pointer to the payload
template<class T>
class messageList {
 public:
  messageList();
  messageList(const messageList& other);
  messageList& operator=(const messageList& other);
  ~messageList();

  /** add data block to list */
  int add(T* headerMsg, T* payloadMsg);
  /** number of data blocks in the list */
  size_t size() {return mDataArray.size();}
  /** clear the list */
  void clear() {mDataArray.clear();}
  /** check if list is empty */
  bool empty() {mDataArray.empty();}

  /**
   * messagePair describes the two sequential messages for header and payload
   * of a data block
   *
   * TODO: decide whether to use pointer to message or pointer to payload in the
   * message. Whith the template approach and appropriate conversion operators in the
   * message class possibly both ways can be served at the same time
   */
  struct messagePair {
    DataHeader_t header;
    T* payload;
  };
  typedef vector<messagePair>::iterator pairIt_t

  class iterator {
   public:
    typedef iterator self_type;
    typedef T value_type;
  
    iterator(pairIt_t& dataIterator) : mDataIterator(dataIterator) { }
    // prefix increment
    self_type operator++() { mDataIterator++; return *this; }
    // postfix increment
    self_type operator++(int unused) {self_type copy(*this); ++*this; return copy;}
    // TODO: given the fact that the data which is hold is a pointer, it always needs to be
    // valid for dereference
    T& operator*() { return *this.operator->();}
    T* operator->() { return (*mDataIterator).payload;}

    bool operator==(const self_type& other) { return mDataIterator == other.mDataIterator; }
    bool operator!=(const self_type& other) { return mDataIterator != other.mDataIterator; }

    bool operator==(const PayloadMetaData_t& md) { return (*mDataIterator).header == md }
    bool operator!=(const PayloadMetaData_t& md) { return (*mDataIterator).header != md }

    /** conversion operator to PayloadMetaData_t struct */
    operator PayloadMetaData_t() {return (*mDataIterator).header;}
    /** return size of payload */
    size_t size() {return (*mDataIterator).header.mPayloadSize;}

  private:
    pairIt_t mDataIterator;
  };

  /** to be defined
  class const_iterator
  {
  };
  */

 private:
  vector<messagePair> mDataArray;
};

template<class T>
int messageList::add(T& headerMsg, T& payloadMsg)
{
  // following conversion relies on the conversion operator for complex types
  uint* headerData = static_cast<uint8_t*>(headerMsg);

  DataHeader_t* srcHeader = reinterpret_cast<DataHeader_t*>(headerData);
  DataHeader_t tgtHeader(*srcHeader);
}

}; // namespace Format
}; // namespace AliceO2
#endif
