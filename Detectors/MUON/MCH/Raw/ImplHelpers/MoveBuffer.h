// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_IMPL_HELPERS_MOVEBUFFER_H
#define O2_MCH_RAW_IMPL_HELPERS_MOVEBUFFER_H

#include <vector>
#include <iostream>
#include <gsl/span>

namespace o2::mch::raw::impl
{

/// Copy the content of b64 to b8
/// Returns the number of bytes copied into b8.
size_t copyBuffer(const std::vector<uint64_t>& b64,
                  std::vector<std::byte>& b8,
                  uint64_t prefix = 0)
{
  constexpr uint64_t m = 0xFF;
  auto s8 = b8.size();
  b8.reserve(s8 + b64.size() / 8);
  for (auto& b : b64) {
    uint64_t g = b | prefix;
    for (uint64_t i = 0; i < 64; i += 8) {
      uint64_t w = m << i;
      b8.emplace_back(std::byte{static_cast<uint8_t>((g & w) >> i)});
    }
  }
  return b8.size() - s8;
}

/// Move the content of b64 to b8 and clears b64.
/// Returns the number of bytes moved into b8.
size_t moveBuffer(std::vector<uint64_t>& b64,
                  std::vector<std::byte>& b8,
                  uint64_t prefix = 0)
{
  auto s = copyBuffer(b64, b8, prefix);
  b64.clear();
  return s;
}

uint64_t b8to64(gsl::span<const std::byte> buffer, size_t i)
{
  return (static_cast<uint64_t>(buffer[i + 0])) |
         (static_cast<uint64_t>(buffer[i + 1]) << 8) |
         (static_cast<uint64_t>(buffer[i + 2]) << 16) |
         (static_cast<uint64_t>(buffer[i + 3]) << 24) |
         (static_cast<uint64_t>(buffer[i + 4]) << 32) |
         (static_cast<uint64_t>(buffer[i + 5]) << 40) |
         (static_cast<uint64_t>(buffer[i + 6]) << 48) |
         (static_cast<uint64_t>(buffer[i + 7]) << 56);
}

/// Copy the content of b8 to b64
/// Returns the number of 64-bits words copied into b64
size_t copyBuffer(gsl::span<const std::byte> b8,
                  std::vector<uint64_t>& b64,
                  uint64_t prefix = 0)
{
  if (b8.size() % 8) {
    throw std::invalid_argument("b8 span must have a size that is a multiple of 8");
  }
  auto s = b64.size();
  for (auto i = 0; i < b8.size(); i += 8) {
    uint64_t w = b8to64(b8, i);
    b64.emplace_back(w | prefix);
  }
  return b64.size() - s;
}

} // namespace o2::mch::raw::impl

#endif
