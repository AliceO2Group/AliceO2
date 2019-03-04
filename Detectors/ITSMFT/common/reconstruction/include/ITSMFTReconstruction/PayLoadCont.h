// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ITSMFT_PAYLOADCONT_H
#define ALICEO2_ITSMFT_PAYLOADCONT_H

#include <cstring>
#include <vector>
#include <functional>
#include <Rtypes.h>

/// \file PayLoadCont.h
/// \brief Declaration of class for continuos buffer of ALPIDE data
/// \author ruben.shahoyan@cern.ch

namespace o2
{
namespace ITSMFT
{

class PayLoadCont
{
  // continous buffer for the payload, just a preallocated vector with current position and end pointer.
  // Big endian is used.

 public:
  static constexpr int MinCapacity = 16;

  ///< allocate buffer
  PayLoadCont(int iniSize = MinCapacity) { expand(iniSize); }
  ~PayLoadCont() = default;

  const uint8_t* data() const { return mBuffer.data(); }

  ///< increase the buffer size
  void expand(int sz);

  bool isEmpty() const { return mPtr == mEnd; }

  ///< make buffer empty w/o deallocating it
  void clear()
  {
    mPtr = mBuffer.data();
    mEnd = mPtr;
  }

  ///< get unused size
  int getUnusedSize() const { return mEnd - mPtr; }

  ///< get filled size
  int getSize() const { return mEnd - mBuffer.data(); }

  ///< get offset of the current ptr from the head
  int getOffset() const { return mPtr - mBuffer.data(); }

  ///< booked capacity
  int getCapacity() const { return mBuffer.size(); }

  ///< number of bytes still can accept w/o expanding the buffer
  int getFreeCapacity() const { return mBuffer.size() - getSize(); }

  ///< make sure buffer may accept at least n bytes
  void ensureFreeCapacity(int n)
  {
    if (getFreeCapacity() < n) {
      expand(getCapacity() + 2 * n);
    }
  }

  ///< add n bytes to the buffer w/o checking for the size
  void addFast(const uint8_t* ptr, int n)
  {
    std::memcpy(mEnd, ptr, n);
    mEnd += n;
  }

  ///< add new byte to the buffer w/o checking for the size
  void addFast(uint8_t val) { *mEnd++ = val; }

  ///< add new short to the buffer w/o checking for the size
  void addFast(uint16_t val)
  {
    *mEnd++ = val >> 8;
    *mEnd++ = 0xff & val;
  }

  ///< erase n bytes w/o checking for the underflow
  void eraseFast(int n) { mEnd -= n; }

  ///< erase n bytes
  void erase(int n)
  {
    if (n > getSize()) {
      clear();
    } else {
      eraseFast(n);
    }
  }

  ///< add n bytes to the buffer, expand if needed. no check for overlap
  void add(const uint8_t* ptr, int n)
  {
    ensureFreeCapacity(n);
    addFast(ptr, n);
  }

  ///< add new byte to the buffer
  void add(uint8_t val)
  {
    ensureFreeCapacity(sizeof(val));
    addFast(val);
  }

  ///< add new short to the buffer
  void add(uint16_t val)
  {
    ensureFreeCapacity(sizeof(val));
    addFast(val);
  }

  ///< direct const access to value at a given slot, w/o checking for overflow
  uint8_t operator[](int i) const { return mBuffer[i]; }

  ///< direct access to value at a given slot, w/o checking for overflow
  uint8_t& operator[](int i) { return mBuffer[i]; }

  ///< read current character value from buffer w/o stepping forward
  bool current(uint8_t& v) const
  {
    if (mPtr < mEnd) {
      v = *mPtr;
      return true;
    }
    return false;
  }

  ///< read character value from buffer
  bool next(uint8_t& v)
  {
    if (mPtr < mEnd) {
      v = *mPtr++;
      return true;
    }
    return false;
  }

  ///< read short value from buffer
  bool next(uint16_t& v)
  {
    if (mPtr < mEnd - (sizeof(uint16_t) - 1)) {
      v = (*mPtr++) << 8;
      v |= (*mPtr++);
      return true;
    }
    return false;
  }

  ///< move current pointer to the head
  void rewind() { mPtr = mBuffer.data(); }

  ///< move all data between the mPtr and mEnd to the head of the buffer
  void moveUnusedToHead()
  {
    auto left = getUnusedSize();
    if (left < getOffset()) {
      std::memcpy(mBuffer.data(), mPtr, left); // there is no overlap
    } else {
      std::memmove(mBuffer.data(), mPtr, left); // there is an overlap
    }
    mPtr = mBuffer.data();
    mEnd = mPtr + left;
  }

  ///< move unused data to the head and upload new chunk of data
  // (attemtint to use all free capacity) using the method provided via getNext
  int append(std::function<int(uint8_t*, int)> getNext)
  {
    moveUnusedToHead();
    auto nRead = getNext(mEnd, getFreeCapacity());
    mEnd += nRead;
    return nRead;
  }

  /// direct write access
  uint8_t* getPtr() { return mPtr; }
  void setPtr(uint8_t* ptr) { mPtr = ptr; }

 private:
  std::vector<uint8_t> mBuffer; //! continuons data buffer
  uint8_t* mPtr = nullptr;      ///! pointer on the position in the buffer
  uint8_t* mEnd = nullptr;      ///! pointer on the last+1 valid entry in the buffer

  ClassDefNV(PayLoadCont, 1);
};
}
}

#endif
