// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUCommonBitSet.h
/// \author David Rohr

#ifndef GPUCOMMONBITSET_H
#define GPUCOMMONBITSET_H

// Limited reimplementation of std::bitset for the GPU.
// Fixed to 32 bits for now.
// In contrast to the GPUCommonArray, we cannot just use std::bitset on the host.
// The layout may be implementation defined, so it is not guarantueed that we
// get correct data after copying it into a gpustd::bitset on the GPU.

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
#include "GPUCommonMath.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include <string>
#endif

namespace o2::gpu::gpustd
{
template <unsigned int N>
class bitset
{
  static_assert(N <= 32, "> 32 bits not supported");

 public:
  GPUdDefault() constexpr bitset() = default;
  GPUdDefault() constexpr bitset(const bitset&) = default;
#ifdef __OPENCL__
  GPUdDefault() constexpr bitset(const __constant bitset&) = default;
#endif // __OPENCL__
  GPUd() constexpr bitset(unsigned int vv) : v(vv){};
  static constexpr unsigned int full_set = ((1ul << N) - 1ul);

  GPUd() constexpr bool all() const { return (v & full_set) == full_set; }
  GPUd() constexpr bool any() const { return v & full_set; }
  GPUd() constexpr bool none() const { return !any(); }

  GPUd() constexpr void set() { v = full_set; }
  GPUd() constexpr void set(unsigned int i) { v |= (1u << i) & full_set; }
  GPUd() constexpr void reset() { v = 0; }
  GPUd() constexpr void reset(unsigned int i) { v &= ~(1u << i); }
  GPUd() constexpr void flip() { v = (~v) & full_set; }

  GPUdDefault() constexpr bitset& operator=(const bitset&) = default;
  GPUd() constexpr bitset operator|(const bitset b) const { return v | b.v; }
  GPUd() constexpr bitset& operator|=(const bitset b)
  {
    v |= b.v;
    return *this;
  }
  GPUd() constexpr bitset operator&(const bitset b) const { return v & b.v; }
  GPUd() constexpr bitset& operator&=(const bitset b)
  {
    v &= b.v;
    return *this;
  }
  GPUd() constexpr bitset operator^(const bitset b) const { return v ^ b.v; }
  GPUd() constexpr bitset& operator^=(const bitset b)
  {
    v ^= b.v;
    return *this;
  }
  GPUd() constexpr bitset operator~() const { return (~v) & full_set; }
  GPUd() constexpr bool operator==(const bitset b) { return v == b.v; }
  GPUd() constexpr bool operator!=(const bitset b) { return v != b.v; }

  GPUd() constexpr bool operator[](unsigned int i) const { return (v >> i) & 1u; }

  GPUd() constexpr unsigned int to_ulong() const { return v; }

  GPUd() constexpr unsigned int count() const
  {
    // count number of non-0 bits in 32bit word
    return GPUCommonMath::Popcount(v);
  }

#ifndef GPUCA_GPUCODE_DEVICE
  std::string to_string() const;
#endif

 private:
  unsigned int v = 0;

  ClassDefNV(bitset, 1);
};

#ifndef GPUCA_GPUCODE_DEVICE
template <unsigned int N>
inline std::string bitset<N>::to_string() const
{
  std::string retVal;
  for (unsigned int i = N; i--;) {
    retVal += std::to_string((int)((*this)[i]));
  }
  return retVal;
}
template <class CharT, class Traits, unsigned int N>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, const bitset<N>& x)
{
  os << x.to_string();
  return os;
}

#endif
} // namespace o2::gpu::gpustd

#endif
