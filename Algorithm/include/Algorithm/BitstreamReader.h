// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef BITSTREAMREADER_H
#define BITSTREAMREADER_H

/// @file   BitstreamReader.h
/// @author Matthias Richter
/// @since  2019-06-05
/// @brief  Utility class to provide bitstream access to an underlying resource

#include <type_traits>
#include <bitset>

namespace o2
{
namespace algorithm
{

/// @class BitStreamReader
/// @brief Utility class to provide bitstream access to an underlying resource
///
/// Allows to access bits of variable length, supports integral types and also
/// bitsets as target type. At the moment, the access is in direction MSB -> LSB.
///
///     BitstreamReader<uint8_t> reader(start, end);
///     while (reader.good() && not reader.eof()) {
///       // get an 8 bit value from the stream, moves the position
///       uint8_t ivalue;
///       reader.get(ivalue);
///
///       // get a 13 bit bitset without moving the position
///       std::bitset<13> value;
///       reader.peek(value, value.size());
///       // e.g. use 7 bits of the data
///       value >>= value.size() - 7;
///       // move position by the specific number of bits
///       reader.seek(7);
///     }
template <typename BufferType>
class BitstreamReader
{
 public:
  using self_type = BitstreamReader<BufferType>;
  // for the moment we simply use pointers, but with some traits this can be extended to
  // containers
  using value_type = BufferType;
  using iterator = const value_type*;
  static constexpr size_t value_size = sizeof(value_type) * 8;
  BitstreamReader() = delete;
  BitstreamReader(iterator start, iterator end)
    : mStart(start), mEnd(end), mCurrent(mStart), mBitPosition(value_size)
  {
  }
  ~BitstreamReader() = default;

  /// Check reader's state
  /// @return true if not in error state
  bool good() const
  {
    return mBitPosition > 0;
  }

  /// Indicates end of data
  /// @return true if end of resource is reached
  bool eof() const
  {
    return mCurrent == mEnd && mBitPosition > 0;
  }

  /// Reset the reader, start over at beginning
  void reset()
  {
    mCurrent = mStart;
    mBitPosition = value_size;
  }

  /// Get the next N bits without moving the read position
  /// if bitlength is smaller than the size of data type, result is aligned to LSB
  /// TODO: this also works nicely for bitsets, but then the bitlength has to be specified
  /// as template parameter, want to do a specific overload, but needs more work to catch
  /// all cases.
  /// @param v  target variable passed by reference
  /// @return number of poked bits
  template <typename T, size_t N = sizeof(T) * 8>
  size_t peek(T& v)
  {
    static_assert(N <= sizeof(T) * 8);
    return peek<T, false>(v, N);
  }

  /// Get the next n bits without moving the read position
  /// if bitlength is smaller than the size of data type, result is aligned to LSB
  /// @param v          target variable passed by reference
  /// @param bitlength  number of bits to read
  /// @return number of poked bits
  template <typename T>
  size_t peek(T& v, size_t bitlength)
  {
    return peek<T, true>(v, bitlength);
  }

  /// Move read position
  /// @param bitlength  move count in number of bits
  void seek(size_t bitlength)
  {
    while (good() && bitlength > 0 && mCurrent != mEnd) {
      if (bitlength >= mBitPosition) {
        bitlength -= mBitPosition;
        mBitPosition = 0;
      } else {
        mBitPosition -= bitlength;
        bitlength = 0;
      }
      if (mBitPosition == 0) {
        mCurrent++;
        mBitPosition = value_size;
      }
    }

    if (bitlength > 0) {
      mBitPosition = 0;
    }
  }

  /// Get the next n bits and move the read position
  template <typename T, size_t N = sizeof(T) * 8>
  T get()
  {
    T result;
    peek<T, N>(result);
    seek(N);
    return result;
  }

  /// Get the next n and move the read position
  template <typename T>
  T get(size_t bitlength = sizeof(T) * 8)
  {
    T result;
    peek<T>(result, bitlength);
    seek(bitlength);
    return result;
  }

  /// @class Bits
  /// @brief Helper class to get value of specified type which holds the number used bits
  ///
  /// The class holds both the extracted value access via peek method and the number of used
  /// bits. The reader will be incremented when the object is destroyed.
  /// The number of bits can be adjusted by using markUsed method
  template <typename FieldType, size_t N = sizeof(FieldType) * 8, typename ParentType = self_type>
  class Bits
  {
   public:
    using field_type = FieldType;
    static_assert(N <= sizeof(FieldType) * 8);
    Bits()
      : mParent(nullptr), mData(0), mLength(0)
    {
    }
    Bits(ParentType* parent, FieldType&& data)
      : mParent(parent), mData(std::move(data)), mLength(N)
    {
    }
    Bits(Bits&& other)
      : mParent(other.mParent), mData(std::move(other.mData)), mLength(other.mLength)
    {
      other.mParent = nullptr;
      other.mLength = 0;
    }

    ~Bits()
    {
      if (mParent) {
        mParent->seek(mLength);
      }
    }

    auto& operator=(Bits<FieldType, N, ParentType>&& other)
    {
      mParent = other.mParent;
      mData = std::move(other.mData);
      mLength = other.mLength;
      other.mParent = nullptr;
      other.mLength = 0;

      return *this;
    }

    FieldType& operator*()
    {
      return mData;
    }

    void markUsed(size_t length)
    {
      mLength = length;
    }

   private:
    ParentType* mParent;
    FieldType mData;
    size_t mLength;
  };

  /// Read an integral value from the stream
  template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
  self_type& operator>>(T& target)
  {
    target = get<T>();
    return *this;
  }

  /// Read a bitstream value from the stream
  template <size_t N>
  self_type& operator>>(std::bitset<N>& target)
  {
    target = get<std::bitset<N>, N>();
    return *this;
  }

  /// Read a Bits object from the stream
  template <typename T>
  self_type& operator>>(Bits<T>& target)
  {
    T bitfield;
    peek<T>(bitfield);
    target = std::move(Bits<T>(this, std::move(bitfield)));
    return *this;
  }

 private:
  /// The internal peek method
  template <typename T, bool RuntimeCheck>
  size_t peek(T& result, size_t bitlength)
  {
    if constexpr (RuntimeCheck) {
      // the runtime check is disabled if bitlength is derived at compile time
      if (bitlength > sizeof(T) * 8) {
        throw std::length_error(std::string("requested bit length ") + std::to_string(bitlength) + " does not fit size of result data type " + std::to_string(sizeof(T) * 8));
      }
    }
    result = 0;
    size_t bitsToWrite = bitlength;
    auto current = mCurrent;
    auto bitsAvailable = mBitPosition;
    while (bitsToWrite > 0 && current != mEnd) {
      // extract available bits
      value_type mask = ~value_type(0) >> (value_size - bitsAvailable);
      if (bitsToWrite >= bitsAvailable) {
        T value = (*current & mask) << (bitsToWrite - bitsAvailable);
        result |= value;
        bitsToWrite -= bitsAvailable;
        bitsAvailable = 0;
      } else {
        value_type value = (*current & mask) >> (bitsAvailable - bitsToWrite);
        result |= value;
        bitsAvailable -= bitsToWrite;
        bitsToWrite = 0;
      }
      if (bitsAvailable == 0) {
        current++;
        bitsAvailable = value_size;
      }
    }

    return bitlength - bitsToWrite;
  }

  /// start of resource
  iterator mStart;
  /// end of resource
  iterator mEnd;
  /// current position in resource
  iterator mCurrent;
  /// bit position in current element
  size_t mBitPosition;
};
} // namespace algorithm
} // namespace o2
#endif
