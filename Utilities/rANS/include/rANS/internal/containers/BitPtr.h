// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   BitPtr.h
/// @author Michael Lettrich
/// @brief  Pointer type helper class for bitwise Packing

#ifndef RANS_INTERNAL_CONTAINERS_BITPTR_H_
#define RANS_INTERNAL_CONTAINERS_BITPTR_H_

#include <cassert>
#include <cstdint>
#include <cstring>

#include <fmt/format.h>

#include "rANS/internal/common/utils.h"

namespace o2::rans
{

class BitPtr
{
 public:
  using bitAddress_type = intptr_t;

  inline constexpr BitPtr() noexcept {};

  inline constexpr BitPtr(intptr_t bitAdr) noexcept : mBitAdr{bitAdr} {};

  template <typename T>
  inline constexpr BitPtr(const T* ptr, intptr_t offset = 0) noexcept : mBitAdr{static_cast<intptr_t>(internal::adr2Bits(ptr)) + offset}
  {
    assert(reinterpret_cast<intptr_t>(ptr) % sizeof(T) == 0); // ensure pointer alignment
  };

  inline constexpr const bitAddress_type& getBitAddress() const noexcept { return mBitAdr; };

  inline constexpr bitAddress_type& getBitAddress() noexcept { return const_cast<bitAddress_type&>(const_cast<const BitPtr&>(*this).getBitAddress()); };

  template <typename T>
  inline constexpr intptr_t getOffset() const noexcept
  {
    assert(mBitAdr >= 0);
    // bit offset from next T
    return mBitAdr % utils::toBits<T>();
  };

  template <typename T>
  inline constexpr T* toPtr() const noexcept
  {
    assert(mBitAdr >= 0);
    // convert from bits to bytes, cutting off offset
    intptr_t byteAdress = getBitAddress();
    if constexpr (sizeof(T) > 1) {
      byteAdress -= getOffset<T>();
    }
    return reinterpret_cast<T*>(byteAdress >> ToBytesShift);
  };

  inline constexpr explicit operator intptr_t() const noexcept
  {
    return getBitAddress();
  };

  template <typename T>
  inline constexpr explicit operator T*() const noexcept
  {
    return toPtr<T>();
  };

  // pointer arithmetics
  inline constexpr BitPtr& operator++() noexcept
  {
    ++mBitAdr;
    return *this;
  };

  inline constexpr BitPtr operator++(int) noexcept
  {
    auto res = *this;
    ++(*this);
    return res;
  };

  inline constexpr BitPtr& operator--() noexcept
  {
    --mBitAdr;
    return *this;
  };

  inline constexpr BitPtr operator--(int) noexcept
  {
    auto res = *this;
    --(*this);
    return res;
  };

  inline constexpr BitPtr& operator+=(intptr_t bitOffset) noexcept
  {
    mBitAdr += bitOffset;
    return *this;
  };

  inline constexpr BitPtr operator+(intptr_t bitOffset) const noexcept
  {
    auto tmp = *const_cast<BitPtr*>(this);
    return tmp += bitOffset;
  }

  inline constexpr BitPtr& operator-=(intptr_t bitOffset) noexcept
  {
    mBitAdr -= bitOffset;
    return *this;
  };

  inline constexpr BitPtr operator-(intptr_t bitOffset) const noexcept
  {
    auto tmp = *const_cast<BitPtr*>(this);
    return tmp -= bitOffset;
  };

  inline constexpr intptr_t operator-(const BitPtr& other) const noexcept
  {
    return this->mBitAdr - other.mBitAdr;
  };

  inline friend BitPtr operator+(intptr_t bitOffset, const BitPtr& bitPtr)
  {
    return bitPtr + bitOffset;
  }

  // comparison
  inline constexpr bool operator==(const BitPtr& other) const noexcept { return this->mBitAdr == other.mBitAdr; };
  inline constexpr bool operator!=(const BitPtr& other) const noexcept { return !(*this == other); };
  inline constexpr bool operator<(const BitPtr& other) const noexcept { return this->mBitAdr < other.mBitAdr; };
  inline constexpr bool operator>(const BitPtr& other) const noexcept { return other < *this; };
  inline constexpr bool operator>=(const BitPtr& other) const noexcept { return !(*this < other); };
  inline constexpr bool operator<=(const BitPtr& other) const noexcept { return !(other < *this); };

  inline friend void swap(BitPtr& first, BitPtr& second)
  {
    using std::swap;
    swap(first.mBitAdr, second.mBitAdr);
  }

 private:
  intptr_t mBitAdr{};
  inline static constexpr intptr_t ToBytesShift = 3;
};

inline std::ostream& operator<<(std::ostream& os, const BitPtr bitPtr)
{
  return os << fmt::format("{:0x}", bitPtr.getBitAddress());
}

} // namespace o2::rans

#endif /* RANS_INTERNAL_CONTAINERS_BITPTR_H_ */