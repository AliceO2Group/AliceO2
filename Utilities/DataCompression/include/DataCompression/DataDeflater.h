//-*- Mode: C++ -*-
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
//  @since  2016-08-08
//  @brief  A general data deflater

#include <cstdint>
#include <cerrno>
#include <stdexcept>
#include <cassert>

namespace o2 {
namespace data_compression {

/**
 * @class CodecIdentity
 * A simple default codec forwarding identity
 */
template<typename CodeType, std::size_t Length = 8 * sizeof(CodeType)>
class CodecIdentity {
public:
  using code_type = CodeType;
  static_assert(Length <= 8 * sizeof(code_type), "CodeType must allow specified bit length");
  static const std::size_t sMaxLength = Length;

  CodecIdentity() = default;
  ~CodecIdentity() = default;

  template<typename ValueType, typename WriterType>
  int write(ValueType v, WriterType writer) {
    code_type code = v;
    return writer(code, sMaxLength);
  }
};

/* TODO: separate the bare deflater and additional functionality like
   bounds check and codec by using a mixin approach
   TODO: error policy, initialization policy
   TODO: bit order policy: MSB or LSB first to buffer
   TODO: monitoring policy
   TODO: bit order: LSB to MSB in every byte or vice versa
 */
template<
  typename TargetType,
  class Codec = CodecIdentity<TargetType>
  >
class DataDeflater {
public:
  using target_type = TargetType;
  static const std::size_t TargetBitWidth = 8 * sizeof(target_type);
  using Writer = std::function<bool(const target_type&)>;

  DataDeflater() : mCurrent(0), mFilledBits(0), mCodec() {}
  ~DataDeflater() {
    // check if the deflater is properly terminated, or pending data will be lost
    assert(mFilledBits == 0);
  }

  /**
   * Reset deflater
   * Drop the current word if a clean start is needed.
   */
  int reset() {
    mCurrent = 0;
    mFilledBits = 0;
  }

  /**
   * Flush and close
   * Write the pending target word
   * @return Number of written words during close
   */
  template<typename WriterT>
  int close(WriterT& writer) {
    int nWords = 0;
    if (mFilledBits > 0) {
      writer(mCurrent);
      ++nWords;
    }
    reset();
    return nWords;
  }

  /**
   * Write number of bits
   * value contains number of valid LSBs given by bitlength
   *
   * TODO: that function might be renamed to simply 'write' in conjunction
   * with a mixin approach. Every deflater mixin instance has only one
   * 'write' function and does internally the necessary conversions to
   * finally use 'write' of the mixin base.
   */
  template <typename ValueType, typename WriterT>
  int writeRaw(ValueType value, uint16_t bitlength, WriterT writer) {
    auto bitsToWrite = bitlength;
    if (bitlength > 8*sizeof(ValueType)) {
      // TODO: error policy
      throw std::runtime_error("bit length exceeds width of the data type");
    }
    while (bitsToWrite > 0) {
      if (mFilledBits == TargetBitWidth) {
        mFilledBits=0;
        writer(mCurrent);
        mCurrent = 0;
      }
      // write at max what is left to be written
      auto writeNow = bitsToWrite;
      // write one element of the target buffer at a time
      if (writeNow > TargetBitWidth) writeNow = TargetBitWidth;
      // write the remaining space in the current element
      auto capacity = TargetBitWidth - mFilledBits;
      if (writeNow > capacity) writeNow = capacity;
      auto mask = (((ValueType)1 << writeNow) - 1) << (bitsToWrite - writeNow);
      auto activebits = (value&mask) >> (bitsToWrite - writeNow);
      mCurrent |= activebits << (capacity-writeNow);
      mFilledBits += writeNow;
      bitsToWrite -= writeNow;
      assert(mFilledBits <= TargetBitWidth);
    }
    return bitlength - bitsToWrite;
  }

  template <typename T, typename WriterT>
  int write(T value, WriterT writer) {
    using RegType = typename Codec::code_type;
    return mCodec.write(value,
                        [&, this] (RegType code, uint16_t codelength) -> int {return this->writeRaw(code, codelength, writer);}
                        );
  }

  /**
   * Align bit output
   * Schedules the write out of the current word at the next occasion
   * (either write or close).
   * @return number of forward bits
   */
  int align() {
    if (mFilledBits == 0 || mFilledBits == TargetBitWidth) return 0;
    // set the number of filled bits to the next target border
    int nBits = TargetBitWidth - mFilledBits;
    mFilledBits = TargetBitWidth;
    return nBits;
  }

private:
  /// current target word
  target_type mCurrent;
  /// current bit position
  int mFilledBits;
  /// codec instance
  Codec mCodec;
};

}; // namespace data_compression
}; // namespace o2

#endif
