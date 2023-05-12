// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_BITSET_H
#define O2_MCH_RAW_BITSET_H

#include <cstdlib>
#include <cstdint>
#include <gsl/span>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace o2
{
namespace mch
{
namespace raw
{

class BitSet
{

 public:
  BitSet();

  // construct a BitSet using a string composed of '0' (meaning
  // bit unset) and '1' (meaning bit set) characters
  // the length of the resulting bitset is that of the string.
  explicit BitSet(std::string_view s);

  ///@{
  // construct a bitset initialized with the x-bits value v
  explicit BitSet(uint8_t v, int n = -1);
  explicit BitSet(uint16_t v, int n = -1);
  explicit BitSet(uint32_t v, int n = -1);
  explicit BitSet(uint64_t v, int n = -1);
  ///@}

 public:
  // check equality
  bool operator==(const BitSet& rhs) const;
  bool operator!=(const BitSet& rhs) const;

  // any returns true if any of the bits is set
  bool any() const;

  // appends a bit at the current position (i.e. len-1)
  int append(bool val);

  ///@{
  // appends the n first bits from a x-bits word.
  // if n is < 0 it is computed for val (using log2(val)+1)
  // otherwise it should be >= log2(val)+1 and <=x
  // and the exact number of specified bits will be set
  // (to 0 or 1).
  // returns the number of bits actually added
  int append(uint8_t val, int n = -1);
  int append(uint16_t val, int n = -1);
  int append(uint32_t val, int n = -1);
  int append(uint64_t val, int n = -1);
  ///@}

  // count returns the number of bits set at 1
  int count() const;

  // sets all the bits to false (i.e. resets)
  void clear();

  // returns true if we hold not bit at all
  bool isEmpty() const { return len() == 0; }

  // sets the value of the bit at given pos
  void set(int pos, bool val);

  // gets the value of the bit at given pos
  bool get(int pos) const;

  // grows the BitSet so it can accomodate at least n bits. Returns true if size changed.
  bool grow(int n);

  // last returns a bitset containing the last n bits of the bitset
  // if there's not enough bits, throw an exception
  BitSet last(int n) const;

  // return the max number of bits this object can currently hold
  // it is a multiple of 8.
  int size() const { return mSize; }

  // return the max number of bits any bitset can hold
  static int maxSize() { return 2 * 8192; }

  // return the number of bits we are current holding
  int len() const { return mLen; }

  // return the max number of bits we've ever held
  int maxlen() const { return mMaxLen; }

  // pruneFirst removes the first n bits from the bitset
  void pruneFirst(int n);

  void setFast(int pos, bool val);

  void setFromBytes(gsl::span<uint8_t> bytes);

  // setRangeFromString populates the bits at indice [a,b] (inclusive range)
  // from the characters in the string: 0 to unset the bit (=false)
  // or 1 to set the bit (=true).
  // A string containing anything else than '0' or '1' is invalid and
  // triggers an exception
  void setRangeFromString(int a, int b, std::string_view s);

  ///@{
  // setRangeFromUint(a,b,uintX_t) populates the bits at indices [a,b] (inclusive range)
  // with the bits of value v. b-a must be <=X otherwise throws an exception
  void setRangeFromUint(int a, int b, uint8_t v);
  void setRangeFromUint(int a, int b, uint16_t v);
  void setRangeFromUint(int a, int b, uint32_t v);
  void setRangeFromUint(int a, int b, uint64_t v);
  ///@}

  // returns a textual representation of the BitSet
  // where the LSB is on the left
  std::string stringLSBLeft() const;

  // returns a textual representation of the BitSet
  // where the LSB is on the right
  std::string stringLSBRight() const;

  // subset returns a subset of the bitset.
  // subset is not a slice (i.e. not a reference, but a copy of the internals)
  // [a,b] inclusive
  BitSet subset(int a, int b) const;

  // uint8 converts the bit set into a 8-bits value, if possible.
  // if b is negative, it is set to the bitset length
  uint8_t uint8(int a, int b) const;

  // uint16 converts the bit set into a 16-bits value, if possible.
  // if b is negative, it is set to the bitset length
  uint16_t uint16(int a, int b) const;

  // uint32 converts the bit set into a 32-bits value, if possible.
  // if b is negative, it is set to the bitset length
  uint32_t uint32(int a, int b) const;

  // uint64 converts the bit set into a 64-bits value, if possible.
  // if b is negative, it is set to the bitset length
  uint64_t uint64(int a, int b) const;

 private:
  int mSize;   // max number of bits we can hold
  int mLen;    // actual number of bits we are holding
  int mMaxLen; // the max number of bits we've ever held
  std::vector<uint8_t> mBytes;
  static int nofInstances;
};

int circularAppend(BitSet& bs, const BitSet& ringBuffer, int startBit, int n);

std::ostream& operator<<(std::ostream& os, const BitSet& bs);

std::string compactString(const BitSet& bs);
} // namespace raw
} // namespace mch
} // namespace o2

#endif
