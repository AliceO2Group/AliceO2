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

#include "DetectorsRaw/RDHUtils.h"
#include "Headers/RAWDataHeader.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawCommon/SampaHeader.h"
#include "MoveBuffer.h"
#include <fmt/format.h>
#include <gsl/span>
#include <iostream>
#include <limits>
#include <vector>

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

inline void append(std::vector<std::byte>& buffer, uint64_t w)
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
void dumpWord(std::ostream& out, uint64_t w);

template <>
void dumpWord<o2::mch::raw::BareFormat>(std::ostream& out, uint64_t w)
{
  // show word itself and it 10-bits parts
  out << fmt::format("{:016X} {:4d} {:4d} {:4d} {:4d} {:4d} ",
                     w,
                     (w & 0x3FF0000000000) >> 40,
                     (w & 0xFFC0000000) >> 30,
                     (w & 0x3FF00000) >> 20,
                     (w & 0xFFC00) >> 10,
                     (w & 0x3FF));
}

template <>
void dumpWord<o2::mch::raw::UserLogicFormat>(std::ostream& out, uint64_t w)
{
  out << fmt::format("{:016X} ", w);
}

template <typename FORMAT, int VERSION = 0>
void dumpWordInfo(std::ostream& out, uint64_t w, const char* spacer = "");

template <>
void dumpWordInfo<o2::mch::raw::BareFormat, 0>(std::ostream& out, uint64_t w, const char* /*spacer*/)
{
  static constexpr uint64_t FIFTYBITSATONE = (static_cast<uint64_t>(1) << 50) - 1;
  SampaHeader h(w & FIFTYBITSATONE);
  if (h == sampaSync()) {
    out << "SYNC !!";
  } else if (h.packetType() == SampaPacketType::Sync) {
    out << "SYNC " << std::boolalpha << (h == sampaSync());
  } else if (h.packetType() == SampaPacketType::Data) {
    out << fmt::format(" n10 {:4d} chip {:2d} ch {:2d}",
                       h.nof10BitWords(), h.chipAddress(), h.channelAddress());
  }
}

template <int VERSION>
void dumpUserLogicWordInfo(std::ostream& out, uint64_t w, const char* spacer)
{
  if (o2::mch::raw::isSampaSync(w)) {
    out << "SYNC ";
  }
  if (w != 0xFEEDDEEDFEEDDEED && w != 0) {
    if (!o2::mch::raw::isSampaSync(w)) {
      out << spacer;
    }
    ULHeaderWord<VERSION> header{w};
    int gbt = header.linkID;
    int elinkid = header.dsID;
    int error = header.error;
    bool incomplete = header.incomplete > 0;
    out << fmt::format("GBT(0.11) {:2d} ELINKID(0..39) {:2d} ERR {:1d} INCOMPLETE {}",
                       gbt, elinkid, error, incomplete);
    if (!o2::mch::raw::isSampaSync(w)) {
      out << fmt::format("{:4d} {:4d} {:4d} {:4d} {:4d} ",
                         (w & 0x3FF0000000000) >> 40,
                         (w & 0xFFC0000000) >> 30,
                         (w & 0x3FF00000) >> 20,
                         (w & 0xFFC00) >> 10,
                         (w & 0x3FF));
    }
  }
}

template <>
void dumpWordInfo<o2::mch::raw::UserLogicFormat, 0>(std::ostream& out, uint64_t w,
                                                    const char* spacer)
{
  dumpUserLogicWordInfo<0>(out, w, spacer);
}

template <>
void dumpWordInfo<o2::mch::raw::UserLogicFormat, 1>(std::ostream& out, uint64_t w,
                                                    const char* spacer)
{
  dumpUserLogicWordInfo<1>(out, w, spacer);
}

template <typename FORMAT, int VERSION>
void dumpBuffer(gsl::span<const std::byte> buffer, std::ostream& out = std::cout, size_t maxbytes = std::numeric_limits<size_t>::max())
{
  int i{0};
  int inRDH{-1};
  const void* rdhP{nullptr};
  const char* spacer = "     ";

  if (buffer.size() < 8) {
    out << "Should at least get 8 bytes to be able to dump\n";
    return;
  }
  while ((i < buffer.size()) && i < maxbytes) {
    if (i % 8 == 0) {
      out << fmt::format("\n{:8d} : ", i);
    }
    uint64_t w = b8to64(buffer, i);
    dumpWord<FORMAT>(out, w);

    if (inRDH >= 0) {
      --inRDH;
    }

    if (buffer.size() >= i + 64) {
      const void* testRDH = reinterpret_cast<const void*>(buffer.data() + i);
      if (o2::raw::RDHUtils::checkRDH(testRDH, false)) {
        rdhP = testRDH;
        inRDH = 8;
        out << "RDH  ";
      }
    }

    if (inRDH == 8) {
      auto version = o2::raw::RDHUtils::getVersion(rdhP);
      if (version >= 6) {
        out << "SOURCE " << o2::raw::RDHUtils::getSourceID(rdhP) << " ";
      }
      out << fmt::format("VER   {:2d} SIZ {:3d} FEEID {:5d}",
                         o2::raw::RDHUtils::getVersion(rdhP),
                         o2::raw::RDHUtils::getHeaderSize(rdhP),
                         o2::raw::RDHUtils::getFEEID(rdhP));
    }

    if (inRDH == 7) {
      std::cout << spacer;
      out << fmt::format("LINK {:3d} CRU {:3d} EP {:1d}",
                         o2::raw::RDHUtils::getLinkID(rdhP),
                         o2::raw::RDHUtils::getCRUID(rdhP),
                         o2::raw::RDHUtils::getEndPointID(rdhP));
    }

    if (inRDH == 6) {
      std::cout << spacer;
      out << fmt::format("ORBIT {:10d} BX {:4d}",
                         o2::raw::RDHUtils::getHeartBeatOrbit(rdhP),
                         o2::raw::RDHUtils::getHeartBeatBC(rdhP));
    }

    if (inRDH == 4) {
      out << spacer;
      out << fmt::format("TRIG  0x{:08X} PAGECOUNT      {:5d}",
                         o2::raw::RDHUtils::getTriggerType(rdhP),
                         o2::raw::RDHUtils::getPageCounter(rdhP));
      if (o2::raw::RDHUtils::getStop(rdhP)) {
        out << " *STOP*";
      }
    }

    if (inRDH == 2) {
      out << spacer;
      out << fmt::format("DET PAR    {:5d} DET FIELD {:10d}",
                         o2::raw::RDHUtils::getDetectorPAR(rdhP),
                         o2::raw::RDHUtils::getDetectorField(rdhP));
    }

    if (inRDH <= 0) {
      dumpWordInfo<FORMAT, VERSION>(out, w, spacer);
    }
    i += 8;
  }
  out << "\n";
}

template <typename FORMAT, int VERSION>
void dumpBuffer(const std::vector<uint64_t>& buffer, std::ostream& out = std::cout, size_t maxbytes = std::numeric_limits<size_t>::max())
{
  std::vector<std::byte> b8;
  for (auto w : buffer) {
    append(b8, w);
  }
  dumpBuffer<FORMAT, VERSION>(b8, out, maxbytes);
}
} // namespace o2::mch::raw::impl
#endif
