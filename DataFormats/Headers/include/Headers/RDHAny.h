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

// @brief placeholder class for arbitraty-version 64B-lonh RDH
// @author ruben.shahoyan@cern.ch

#ifndef ALICEO2_HEADER_RDHANY_H
#define ALICEO2_HEADER_RDHANY_H
#include "GPUCommonDef.h"
#include "Headers/RAWDataHeader.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include <type_traits>
#include <stdexcept>
#endif

namespace o2
{
namespace header
{

struct RDHAny {
  uint64_t word0 = 0x0;
  uint64_t word1 = 0x0;
  uint64_t word2 = 0x0;
  uint64_t word3 = 0x0;
  uint64_t word4 = 0x0;
  uint64_t word5 = 0x0;
  uint64_t word6 = 0x0;
  uint64_t word7 = 0x0;

  RDHAny(int v = 0); // 0 for default version

  template <typename H>
  RDHAny(const H& rdh);

  template <typename H>
  RDHAny& operator=(const H& rdh);

  //------------------ service methods
  using RDHv4 = o2::header::RAWDataHeaderV4; // V3 == V4
  using RDHv5 = o2::header::RAWDataHeaderV5;
  using RDHv6 = o2::header::RAWDataHeaderV6;
  using RDHv7 = o2::header::RAWDataHeaderV7; // update this for every new version

  /// make sure we RDH is a legitimate RAWDataHeader
  template <typename RDH>
  GPUhdi() static constexpr void sanityCheckStrict()
  {
#ifndef GPUCA_GPUCODE_DEVICE
    static_assert(std::is_same<RDH, RDHv4>::value || std::is_same<RDH, RDHv5>::value ||
                    std::is_same<RDH, RDHv6>::value || std::is_same<RDH, RDHv7>::value,
                  "not an RDH");
#endif
  }

  /// make sure we RDH is a legitimate RAWDataHeader or generic RDHAny placeholder
  template <typename RDH>
  GPUhdi() static constexpr void sanityCheckLoose()
  {
#ifndef GPUCA_GPUCODE_DEVICE
    static_assert(std::is_same<RDH, RDHv4>::value || std::is_same<RDH, RDHv5>::value ||
                    std::is_same<RDH, RDHv6>::value || std::is_same<RDH, RDHv7>::value || std::is_same<RDHAny, RDH>::value,
                  "not an RDH or RDHAny");
#endif
  }

  template <typename H>
  GPUhdi() static const void* voidify(const H& rdh)
  {
    sanityCheckLoose<H>();
    return reinterpret_cast<const void*>(&rdh);
  }

  template <typename H>
  GPUhdi() static void* voidify(H& rdh)
  {
    sanityCheckLoose<H>();
    return reinterpret_cast<void*>(&rdh);
  }

  GPUhdi() const void* voidify() const { return voidify(*this); }
  GPUhdi() void* voidify() { return voidify(*this); }

  template <typename H>
  GPUhdi() H* as_ptr()
  {
    sanityCheckLoose<H>();
    return reinterpret_cast<H*>(this);
  }

  template <typename H>
  GPUhdi() H& as_ref()
  {
    sanityCheckLoose<H>();
    return reinterpret_cast<H&>(this);
  }

 protected:
  void copyFrom(const void* rdh);
};

///_________________________________
/// create from arbitrary RDH version
template <typename H>
inline RDHAny::RDHAny(const H& rdh)
{
  sanityCheckLoose<H>();
  copyFrom(&rdh);
}

///_________________________________
/// copy from arbitrary RDH version
template <typename H>
inline RDHAny& RDHAny::operator=(const H& rdh)
{
  sanityCheckLoose<H>();
  if (this != voidify(rdh)) {
    copyFrom(&rdh);
  }
  return *this;
}

} // namespace header
} // namespace o2

#endif
