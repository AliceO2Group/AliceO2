// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_IMPL_HELPERS_DUMPBUFFER_H
#define O2_MCH_RAW_IMPL_HELPERS_DUMPBUFFER_H

#include <gsl/span>
#include <iostream>
#include <fmt/format.h>
#include <vector>
#include "MCHRawCommon/SampaHeader.h"
#include <limits>
#include "MCHRawCommon/RDHManip.h"
#include "Headers/RAWDataHeader.h"

namespace o2::mch::raw::impl
{

template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 1>
void dumpBuffer(gsl::span<T> buffer)
{

  int i{0};
  while (i < buffer.size()) {
    if (i % (16 / sizeof(T)) == 0) {
      std::cout << fmt::format("\n{:8d} : ", i * sizeof(T));
    }
    std::cout << fmt::format("{:0{}X} ", buffer[i], sizeof(T) * 2);
    i++;
  }
  std::cout << "\n";
}

uint64_t b8to64(gsl::span<uint8_t> buffer, size_t i)
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

void append(std::vector<uint8_t>& buffer, uint64_t w)
{
  buffer.emplace_back(static_cast<uint8_t>((w & UINT64_C(0x00000000000000FF))));
  buffer.emplace_back(static_cast<uint8_t>((w & UINT64_C(0x000000000000FF00)) >> 8));
  buffer.emplace_back(static_cast<uint8_t>((w & UINT64_C(0x0000000000FF0000)) >> 16));
  buffer.emplace_back(static_cast<uint8_t>((w & UINT64_C(0x00000000FF000000)) >> 24));
  buffer.emplace_back(static_cast<uint8_t>((w & UINT64_C(0x000000FF00000000)) >> 32));
  buffer.emplace_back(static_cast<uint8_t>((w & UINT64_C(0x0000FF0000000000)) >> 40));
  buffer.emplace_back(static_cast<uint8_t>((w & UINT64_C(0x00FF000000000000)) >> 48));
  buffer.emplace_back(static_cast<uint8_t>((w & UINT64_C(0xFF00000000000000)) >> 56));
}

void dumpBuffer(gsl::span<uint8_t> buffer, std::ostream& out = std::cout, size_t maxbytes = std::numeric_limits<size_t>::max())
{
  int i{0};
  int inRDH{0};

  if (buffer.size() < 8) {
    std::cout << "Should at least get 8 bytes to be able to dump\n";
    return;
  }
  while ((i < buffer.size() - 7) && i < maxbytes) {
    if (i % 8 == 0) {
      out << fmt::format("\n{:8d} : ", i);
    }
    uint64_t w = b8to64(buffer, i);
    out << fmt::format("{:016X} {:4d} {:4d} {:4d} {:4d} {:4d} ",
                       w,
                       (w & 0x3FF0000000000) >> 40,
                       (w & 0xFFC0000000) >> 30,
                       (w & 0x3FF00000) >> 20,
                       (w & 0xFFC00) >> 10,
                       (w & 0x3FF));
    if ((w & 0xFFFF) == 0x4004) {
      inRDH = 8;
    }
    if (inRDH) {
      --inRDH;
      if (inRDH == 7) {
        out << "Begin RDH ";
        auto const rdh = o2::mch::raw::createRDH<o2::header::RAWDataHeaderV4>(buffer.subspan(i, 64));
        std::cout << fmt::format("ORBIT {} BX {} FEEID {}", rdhOrbit(rdh), rdhBunchCrossing(rdh),
                                 rdh.feeId);
      }
      if (inRDH == 0) {
        out << "End RDH ";
      }
    } else {
      SampaHeader h(w & 0x3FFFFFFFFFFFF);
      if (h.packetType() == SampaPacketType::Sync) {
        out << "SYNC";
      } else if (h.packetType() == SampaPacketType::Data) {
        out << fmt::format(" n10 {:4d} chip {:2d} ch {:2d}",
                           h.nof10BitWords(), h.chipAddress(), h.channelAddress());
      }
    }
    i += 8;
  }
  out << "\n";
}
void dumpBuffer(const std::vector<uint64_t>& buffer, std::ostream& out = std::cout, size_t maxbytes = std::numeric_limits<size_t>::max())
{
  std::vector<uint8_t> b8;
  for (auto w : buffer) {
    append(b8, w);
  }
  dumpBuffer(b8, out, maxbytes);
}

} // namespace o2::mch::raw::impl
#endif
