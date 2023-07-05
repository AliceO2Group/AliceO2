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

#include "CommonConstants/Triggers.h"
#include "Headers/RAWDataHeader.h"
#include "RDHManip.h"
#include <cassert>
#include <fmt/format.h>
#include <iostream>
#include <string>
#include "CommonConstants/Triggers.h"
#include "Headers/RDHAny.h"
#include "DetectorsRaw/RDHUtils.h"

namespace
{
std::string triggerTypeAsString(uint32_t triggerType)
{
  std::string s;

  if (triggerType == 0) {
    s = "UNKNOWN";
  }

  if (triggerType & o2::trigger::ORBIT) {
    s += "ORBIT ";
  }
  if (triggerType & o2::trigger::HB) {
    s += "HB ";
  }
  if (triggerType & o2::trigger::HBr) {
    s += "HBr ";
  }
  if (triggerType & o2::trigger::HC) {
    s += "HC ";
  }
  if (triggerType & o2::trigger::PhT) {
    s += "PhT ";
  }
  if (triggerType & o2::trigger::PP) {
    s += "PP ";
  }
  if (triggerType & o2::trigger::Cal) {
    s += "Cal ";
  }
  if (triggerType & o2::trigger::SOT) {
    s += "SOT ";
  }
  if (triggerType & o2::trigger::EOT) {
    s += "EOT ";
  }
  if (triggerType & o2::trigger::SOC) {
    s += "SOC ";
  }
  if (triggerType & o2::trigger::EOC) {
    s += "EOC ";
  }
  if (triggerType & o2::trigger::TF) {
    s += "TF ";
  }
  if (triggerType & o2::trigger::TPC) {
    s += "TPC ";
  }
  if (triggerType & o2::trigger::TPCrst) {
    s += "TPCrst ";
  }
  if (triggerType & o2::trigger::TOF) {
    s += "TOF ";
  }
  return s;
}
} // namespace

namespace o2::header
{
std::ostream& operator<<(std::ostream& os, const o2::header::RDHAny& rdh)
{
  auto triggerType = o2::raw::RDHUtils::getTriggerType(rdh);
  auto headerSize = o2::raw::RDHUtils::getHeaderSize(rdh);

  os << fmt::format("version              {:03d} headerSize      {:03d} triggerType {:08x} {:s}\n",
                    o2::raw::RDHUtils::getVersion(rdh),
                    headerSize,
                    triggerType,
                    triggerTypeAsString(triggerType));

  os << fmt::format("cruId                {:03d} dpwId            {:02d} linkId           {:03d}\n",
                    o2::raw::RDHUtils::getCRUID(rdh),
                    o2::raw::RDHUtils::getEndPointID(rdh),
                    o2::raw::RDHUtils::getLinkID(rdh));

  auto memorySize = o2::raw::RDHUtils::getMemorySize(rdh);

  os << fmt::format("offsetToNext       {:05d} memorySize    {:05d}                      {:s}\n",
                    o2::raw::RDHUtils::getOffsetToNext(rdh),
                    memorySize,
                    memorySize == headerSize ? "EMPTY" : "");

  os << fmt::format("heartbeatOrbit{:010d} heartbeatBC    {:04d} feeId         {:6d}\n",
                    o2::raw::RDHUtils::getHeartBeatOrbit(rdh),
                    o2::raw::RDHUtils::getHeartBeatBC(rdh),
                    o2::raw::RDHUtils::getFEEID(rdh));

  auto stop = o2::raw::RDHUtils::getStop(rdh);
  os << fmt::format("stopBit                {:1d} pagesCounter    {:03d} packetCounter    {:03d} {:s}\n",
                    stop,
                    o2::raw::RDHUtils::getPageCounter(rdh),
                    o2::raw::RDHUtils::getPacketCounter(rdh),
                    stop ? "STOP" : "");

  return os;
}

} // namespace o2::header

using namespace o2::header;

namespace o2::mch::raw
{

void append(std::vector<std::byte>& buffer, uint64_t w)
{
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x00000000000000FF)) >> 0)});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x000000000000FF00)) >> 8)});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x0000000000FF0000)) >> 16)});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x00000000FF000000)) >> 24)});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x000000FF00000000)) >> 32)});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x0000FF0000000000)) >> 40)});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0x00FF000000000000)) >> 48)});
  buffer.emplace_back(std::byte{static_cast<uint8_t>((w & UINT64_C(0xFF00000000000000)) >> 56)});
}

void append(std::vector<uint32_t>& buffer, uint64_t w)
{
  buffer.emplace_back(static_cast<uint32_t>(w & 0xFFFFFFFF));
  buffer.emplace_back(static_cast<uint32_t>((w & UINT64_C(0xFFFFFFFF00000000)) >> 32));
}

void appendRDH(std::vector<std::byte>& buffer, const o2::header::RDHAny& rdh)
{
  append(buffer, rdh.word0);
  append(buffer, rdh.word1);
  append(buffer, rdh.word2);
  append(buffer, rdh.word3);
  append(buffer, rdh.word4);
  append(buffer, rdh.word5);
  append(buffer, rdh.word6);
  append(buffer, rdh.word7);
}

uint64_t eightBytes(gsl::span<const std::byte> buffer)
{
  return (static_cast<uint64_t>(buffer[0])) |
         (static_cast<uint64_t>(buffer[1]) << 8) |
         (static_cast<uint64_t>(buffer[2]) << 16) |
         (static_cast<uint64_t>(buffer[3]) << 24) |
         (static_cast<uint64_t>(buffer[4]) << 32) |
         (static_cast<uint64_t>(buffer[5]) << 40) |
         (static_cast<uint64_t>(buffer[6]) << 48) |
         (static_cast<uint64_t>(buffer[7]) << 56);
}

o2::header::RDHAny createRDH(gsl::span<const std::byte> buffer, int version)
{
  auto minSize = sizeof(o2::header::RDHAny);
  if (buffer.size() < minSize) {
    throw std::invalid_argument(fmt::format("buffer should be at least {} bytes", minSize));
  }
  o2::header::RDHAny rdh;

  o2::raw::RDHUtils::setVersion(rdh, version);

  rdh.word0 = eightBytes(buffer.subspan(0));
  rdh.word1 = eightBytes(buffer.subspan(8));
  rdh.word2 = eightBytes(buffer.subspan(16));
  rdh.word3 = eightBytes(buffer.subspan(24));
  rdh.word4 = eightBytes(buffer.subspan(32));
  rdh.word5 = eightBytes(buffer.subspan(40));
  rdh.word6 = eightBytes(buffer.subspan(48));
  rdh.word7 = eightBytes(buffer.subspan(56));

  return rdh;
}

int forEachRDH(gsl::span<const std::byte> buffer, std::function<void(const void*)> f = nullptr)
{
  int index{0};
  int nrdh{0};
  while (index < buffer.size()) {
    const void* rdhPtr = reinterpret_cast<const void*>(buffer.data() + index);
    if (!o2::raw::RDHUtils::checkRDH(rdhPtr, true)) {
      break;
    }
    nrdh++;
    if (f) {
      f(rdhPtr);
    }
    auto offsetToNext = o2::raw::RDHUtils::getOffsetToNext(rdhPtr);
    if (offsetToNext == 0) {
      return -1;
    }
    index += offsetToNext;
  }
  return nrdh;
}

int showRDHs(gsl::span<const std::byte> buffer)
{
  assert(buffer.size() % 4 == 0);
  return forEachRDH(buffer, [](const void* rdhPtr) {
    auto version = o2::raw::RDHUtils::getVersion(rdhPtr);
    switch (version) {
      case 3:
      case 4:
        std::cout << (*reinterpret_cast<const RAWDataHeaderV4*>(rdhPtr)) << "\n";
        break;
      case 5:
        std::cout << (*reinterpret_cast<const RAWDataHeaderV5*>(rdhPtr)) << "\n";
        break;
      case 6:
        std::cout << (*reinterpret_cast<const RAWDataHeaderV6*>(rdhPtr)) << "\n";
        break;
      case 7:
        std::cout << (*reinterpret_cast<const RAWDataHeaderV7*>(rdhPtr)) << "\n";
        break;
      default:
        throw std::invalid_argument(fmt::format("RDH version {} not yet supported by showRDHs function",
                                                version));
    }
  });
}

int countRDHs(gsl::span<const std::byte> buffer)
{
  return forEachRDH(buffer);
}

} // namespace o2::mch::raw
