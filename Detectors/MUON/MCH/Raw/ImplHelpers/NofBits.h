// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_IMPL_HELPERS_NOFBITS_H
#define O2_MCH_RAW_IMPL_HELPERS_NOFBITS_H

#include <cstdlib>
#include <string_view>
#include <cmath>
#include <fmt/format.h>

namespace o2::mch::raw::impl
{
template <typename T>
int nofBits(T val)
{
  return static_cast<int>(std::floor(log2(1.0 * val)) + 1);
}

template <typename T>
void assertNofBits(std::string_view msg, T value, int n)
{
  // throws an exception if value is not contained within n bits
  if (static_cast<uint64_t>(value) >= (static_cast<uint64_t>(1) << n)) {
    throw std::invalid_argument(fmt::format("{} : 0x{:x} has {} bits, which is more than the {} allowed", msg, value, nofBits(value), n));
  }
}
} // namespace o2::mch::raw::impl

#endif
