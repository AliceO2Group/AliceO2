//-*- Mode: C++ -*-

#ifndef DATADEFLATER_H
#define DATADEFLATER_H
//****************************************************************************
//* This file is free software: you can redistribute it and/or modify        *
//* it under the terms of the GNU General Public License as published by     *
//* the Free Software Foundation, either version 3 of the License, or        *
//* (at your option) any later version.                                      *
//*                                                                          *
//* Primary Authors: Matthias Richter <richterm@scieq.net>                   *
//*                                                                          *
//* The authors make no claims about the suitability of this software for    *
//* any purpose. It is provided "as is" without express or implied warranty. *
//****************************************************************************

//  @file   DataDeflater.h
//  @author Matthias Richter
//  @since  2015-08-08
//  @brief  A general data deflater

#include <cstdint>
#include <cerrno>

namespace AliceO2 {

/* TODO: separate the bare deflater and additional functionality like
   bounds check and codec by using a mixin approach
   TODO: error policy, initialization policy
   TODO: bit order policy: MSB or LSB first to buffer
   TODO: monitoring policy
   TODO: bit order: LSB to MSB in every byte or vice versa
 */

template<
  typename _RegType,
  typename _TargetType,
  //  typename BoundsCheck, // bounds check policy
  class Codec
  >
class DataDeflater {
 public:
  DataDeflater() : mBegin(nullptr), mEnd(nullptr), mCurrent(mEnd), mBitPosition(0), mCodec() {}
  ~DataDeflater() {}

  /**
   * Init target
   * TODO: think about other targets than a buffer
   */
  int Init(_TargetType* buffer, int size) {
    // check if no active buffer according to error policy
    mBegin = buffer;
    mEnd = mBegin + size;
    mCurrent = mBegin;
    mBitPosition = 0;
  }

  /**
   * Flush and close, invalidate output target
   *
   * @return Number of written elements
   */
  int Close() {
    if (mBitPosition > 0) mCurrent++;
    int nElements = mCurrent - mBegin;
    mBegin = nullptr;
    mEnd = mBegin;
    mCurrent = mEnd;
    mBitPosition = 0;
    return nElements;
  }

  /**
   * Write number of bits
   * value contains number of valid LSBs given by bitlength
   *
   * TODO: that function might be renamed to simply 'Write' in conjunction
   * with a mixin approach. Every deflater mixin instance has only one
   * 'Write' function and does internally the necessary conversions to
   * finally use 'Write' of the mixin base.
   */
  template <typename ValueType>
  int WriteRaw(ValueType value, uint16_t bitlength) {
    uint16_t bitsWritten = 0;
    if (bitlength > 8*sizeof(ValueType)) {
      // TODO: error policy
      bitlength = 8*sizeof(ValueType);
    }
    while (bitsWritten < bitlength) {
      if (mCurrent == mEnd) {
        //break; // depending on error policy
        return -ENOSPC;
      }
      _TargetType& current = *mCurrent;
      // write at max what is left to be written
      uint16_t writeNow = bitlength - bitsWritten;
      ValueType mask = 1<<(writeNow); mask -= 1;
      // write one element of the target buffer at a time
      if (writeNow > 8*sizeof(_TargetType)) writeNow = 8*sizeof(_TargetType);
      // write the remaining space in the current element
      uint16_t capacity=8*sizeof(_TargetType)-mBitPosition;
      if (writeNow > capacity) writeNow = capacity;
      ValueType activebits=(value&mask)>>(bitlength-bitsWritten-writeNow);
      activebits<<(capacity-writeNow);
      mBitPosition+=writeNow;
      bitsWritten+=writeNow;
      if (mBitPosition==8*sizeof(_TargetType)) {
        mBitPosition=0;
        mCurrent++;
      } // pedantic check: should never exceed the taget type size
    }
    return bitsWritten;
  }

  int WriteRaw(bool bit) {
    return WriteRaw(bit, 1);
  }

  template <typename T>
  int Write(T value) {
    return mCodec.Write(value, _RegType(0),
                         [this] (_RegType value, uint16_t bitlength) -> int {return this->WriteRaw(value, bitlength);}
                         );
  }

  /**
   * Align bit output
   * @return number of forward bits
   */
  int Align() {
    if (mBitPosition == 0 || mCurrent == mEnd) return 0;
    int nforward = sizeof(_TargetType) - mBitPosition;
    mBitPosition = 0;
    mCurrent++;
    return nforward;
  }

  void print() {
    int bufferSize = mEnd - mBegin;
    int filledSize = mCurrent - mBegin;
    std::cout << "DataDeflater: " << bufferSize << " elements of bit width " << 8*sizeof(_TargetType) << std::endl;
    if (bufferSize > 0)
      std::cout << "    position: " << filledSize << " (bit " << mBitPosition << ")" << std::endl;
  }

 private:
  /// start of write target
  _TargetType* mBegin;
  /// end of write target: pointer to just after target
  _TargetType* mEnd;
  /// current target position
  _TargetType* mCurrent;
  /// current bit position
  int           mBitPosition;
  /// codec instance
  Codec         mCodec;

};

}; // namespace AliceO2

#endif
