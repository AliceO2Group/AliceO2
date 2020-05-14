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
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawElecMap/DsElecId.h"
#include "MoveBuffer.h"

namespace o2::mch::raw::impl
{

template <typename T, std::enable_if_t<std::is_integral<T>::value, int> = 1>
void dumpByteBuffer(gsl::span<T> buffer)
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

void append(std::vector<std::byte>& buffer, uint64_t w)
{
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x00000000000000FF)))});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x000000000000FF00)) >> 8)});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x0000000000FF0000)) >> 16)});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x00000000FF000000)) >> 24)});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x000000FF00000000)) >> 32)});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x0000FF0000000000)) >> 40)});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x00FF000000000000)) >> 48)});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0xFF00000000000000)) >> 56)});
}

template <typename FORMAT>
void dumpBuffer(gsl::span<const std::byte> buffer, std::ostream& out = std::cout, size_t maxbytes = std::numeric_limits<size_t>::max());

template <>
void dumpBuffer<o2::mch::raw::BareFormat>(gsl::span<const std::byte> buffer, std::ostream& out, size_t maxbytes)
{
  int i{0};
  int inRDH{0};

  if (buffer.size() < 8) {
    std::cout << "Should at least get 8 bytes to be able to dump\n";
    return;
  }
  while ((i < buffer.size()) && i < maxbytes) {
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
      constexpr uint64_t FIFTYBITSATONE = (static_cast<uint64_t>(1) << 50) - 1;
      SampaHeader h(w & FIFTYBITSATONE);
      std::cout << "14first=" << ((w & 0xFFFC000000000000) >> 50) << " ";
      if (h == sampaSync()) {
        out << "SYNC !!";
      } else if (h.packetType() == SampaPacketType::Sync) {
        out << "SYNC " << std::boolalpha << (h == sampaSync());
      } else if (h.packetType() == SampaPacketType::Data) {
        out << fmt::format(" n10 {:4d} chip {:2d} ch {:2d}",
                           h.nof10BitWords(), h.chipAddress(), h.channelAddress());
      }
    }
    i += 8;
  }
  out << "\n";
}

template <>
void dumpBuffer<o2::mch::raw::UserLogicFormat>(gsl::span<const std::byte> buffer, std::ostream& out, size_t maxbytes)
{
  int i{0};
  int inRDH{-1};
  o2::header::RAWDataHeaderV4 rdh;

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
    if (inRDH >= 0) {
      --inRDH;
    }

    if (buffer.size() >= i + 64) {
      rdh = o2::mch::raw::createRDH<o2::header::RAWDataHeaderV4>(buffer.subspan(i, 64));
      if (isValid(rdh)) {
        inRDH = 8;
        out << "Begin RDH ";
        std::cout << fmt::format("ORBIT {} BX {} PAYLOADSIZE {}", rdhOrbit(rdh), rdhBunchCrossing(rdh), rdhPayloadSize(rdh));
      }
    }
    if (inRDH <= 0) {
      constexpr uint64_t FIFTYBITSATONE = (static_cast<uint64_t>(1) << 50) - 1;
      SampaHeader h(w & FIFTYBITSATONE);
      if (h == sampaSync()) {
        out << "SYNC --- ";
      }

      std::cout << fmt::format("GBT(0.11) {:2d} ELINKID(0..39) {:4d} ERR {:2d}",
                               (w >> 59) & 0x1F,
                               (w >> 53) & 0x3F,
                               (w >> 50) & 0x7);
    }

    i += 8;
  }
  out << "\n";
} // namespace o2::mch::raw::impl

template <typename FORMAT>
void dumpBuffer(const std::vector<uint64_t>& buffer, std::ostream& out = std::cout, size_t maxbytes = std::numeric_limits<size_t>::max())
{
  std::vector<std::byte> b8;
  for (auto w : buffer) {
    append(b8, w);
  }
  dumpBuffer<FORMAT>(b8, out, maxbytes);
}

} // namespace o2::mch::raw::impl
#endif
