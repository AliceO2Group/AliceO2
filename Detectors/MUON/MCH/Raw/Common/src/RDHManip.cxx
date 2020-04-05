// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CommonConstants/Triggers.h"
#include "Headers/RAWDataHeader.h"
#include "MCHRawCommon/RDHManip.h"
#include <cassert>
#include <fmt/format.h>
#include <iostream>
#include <string>
#include "CommonConstants/Triggers.h"

namespace o2::header
{
std::ostream& operator<<(std::ostream& os, const o2::header::RAWDataHeaderV4& rdh)
{
  os << fmt::format("version              {:03d} headerSize      {:03d} triggerType {:08x} {:s}\n",
                    rdh.version,
                    rdh.headerSize,
                    rdh.triggerType,
                    o2::mch::raw::triggerTypeAsString(rdh.triggerType));

  os << fmt::format("cruId                {:03d} dpwId            {:02d} linkId           {:03d}\n", rdh.cruID, rdh.endPointID, rdh.linkID);

  os << fmt::format("offsetToNext       {:05d} memorySize    {:05d} blockLength    {:05d} {:s}\n", rdh.offsetToNext, rdh.memorySize, rdh.blockLength,
                    rdh.memorySize == sizeof(rdh) ? "EMPTY" : "");

  os << fmt::format("heartbeatOrbit{:010d} heartbeatBC    {:04d} feeId         {:6d}\n",
                    rdh.heartbeatOrbit, rdh.heartbeatBC, rdh.feeId);

  os << fmt::format("stopBit                {:1d} pagesCounter    {:03d} packetCounter    {:03d} {:s}\n",
                    rdh.stop, rdh.pageCnt, rdh.packetCounter,
                    rdh.stop ? "STOP" : "");

  return os;
}

} // namespace o2::header

using namespace o2::header;

namespace o2::mch::raw
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

template <>
bool isValid(const RAWDataHeaderV4& rdh)
{
  return rdh.version == 4 && rdh.headerSize == 64;
}

template <>
void assertRDH(const RAWDataHeaderV4& rdh)
{
  if (rdh.version != 4) {
    throw std::invalid_argument(fmt::format("RDH version {} is not the expected 4",
                                            rdh.version));
  }
  if (rdh.headerSize != 64) {
    throw std::invalid_argument(fmt::format("RDH size {} is not the expected 64",
                                            rdh.headerSize));
  }
}

void append(std::vector<uint32_t>& buffer, uint64_t w)
{
  buffer.emplace_back(static_cast<uint32_t>(w & 0xFFFFFFFF));
  buffer.emplace_back(static_cast<uint32_t>((w & UINT64_C(0xFFFFFFFF00000000)) >> 32));
}

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

template <>
void appendRDH(std::vector<std::byte>& buffer, const RAWDataHeaderV4& rdh)
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

template <>
RAWDataHeaderV4 createRDH(gsl::span<const std::byte> buffer)
{
  if (buffer.size() < 64) {
    throw std::invalid_argument("buffer should be at least 64 bytes");
  }
  RAWDataHeaderV4 rdh;
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

template <>
RAWDataHeaderV4 createRDH(uint16_t cruId, uint8_t linkId, uint16_t feeId, uint32_t orbit, uint16_t bunchCrossing,
                          uint16_t payloadSize)
{
  RAWDataHeaderV4 rdh;

  if (payloadSize > 0x0FFFF - sizeof(rdh)) {
    throw std::invalid_argument("payload too big");
  }

  uint16_t memorySize = payloadSize + sizeof(rdh);

  rdh.cruID = cruId;
  rdhLinkId(rdh, linkId);
  rdh.feeId = feeId;
  rdh.priority = 0;
  rdh.blockLength = memorySize - sizeof(rdh); // FIXME: the blockLength disappears in RDHv5 ?
  rdh.memorySize = memorySize;
  rdh.offsetToNext = memorySize;
  rdh.packetCounter = 0;
  rdh.triggerType = 0;
  rdh.detectorField = 0; // FIXME: fill this ?
  rdh.par = 0;           // FIXME: fill this ?
  rdh.stop = 0;
  rdh.pageCnt = 0;
  rdh.heartbeatOrbit = orbit;
  rdh.heartbeatBC = bunchCrossing;

  return rdh;
}

template <typename RDH>
int showRDHs(gsl::span<const std::byte> buffer)
{
  assert(buffer.size() % 4 == 0);
  return forEachRDH<RDH>(buffer, [](const RDH& rdh) {
    std::cout << rdh << "\n";
  });
}

template <typename RDH>
int countRDHs(gsl::span<const std::byte> buffer)
{
  return forEachRDH<RDH>(buffer, static_cast<std::function<void(const RDH&)>>(nullptr));
}

template <typename RDH>
int forEachRDH(gsl::span<uint8_t> buffer, std::function<void(RDH&, gsl::span<uint8_t>::size_type offset)> f)
{
  int index{0};
  int nrdh{0};
  while (index < buffer.size()) {
    RDH* rdhPtr = reinterpret_cast<RDH*>(&buffer[index]);
    RDH& rdh = *rdhPtr;
    if (!isValid(rdh)) {
      break;
    }
    nrdh++;
    if (f) {
      f(rdh, index);
    }
    if (rdh.offsetToNext == 0) {
      return -1;
    }
    index += rdh.offsetToNext;
  }
  return nrdh;
}

template <typename RDH>
int forEachRDH(gsl::span<const std::byte> buffer, std::function<void(const RDH&)> f)
{
  assert(buffer.size() % 4 == 0);
  RDH rdh;
  int index{0};
  int nrdh{0};
  while (index < buffer.size()) {
    memcpy(&rdh, &buffer[index], sizeof(rdh));
    if (!isValid(rdh)) {
      break;
    }
    nrdh++;
    if (f) {
      f(rdh);
    }
    if (rdh.offsetToNext == 0) {
      return -1;
    }
    index += rdh.offsetToNext;
  }
  return nrdh;
}

template <typename RDH>
int forEachRDH(gsl::span<const std::byte> buffer, std::function<void(const RDH&, gsl::span<const std::byte>::size_type)> f)
{
  assert(buffer.size() % 4 == 0);
  RDH rdh;
  int index{0};
  int nrdh{0};
  while (index < buffer.size()) {
    memcpy(&rdh, &buffer[index], sizeof(rdh));
    if (!isValid(rdh)) {
      break;
    }
    nrdh++;
    if (f) {
      f(rdh, index);
    }
    if (rdh.offsetToNext == 0) {
      return -1;
    }
    index += rdh.offsetToNext;
  }
  return nrdh;
}

// force instanciation of (only) the templates we need

// RDH v4 versions

template int showRDHs<RAWDataHeaderV4>(gsl::span<const std::byte> buffer);
template int countRDHs<RAWDataHeaderV4>(gsl::span<const std::byte> buffer);
template int forEachRDH(gsl::span<const std::byte> buffer, std::function<void(const RAWDataHeaderV4&)> f);
template int forEachRDH(gsl::span<uint8_t> buffer, std::function<void(RAWDataHeaderV4&, gsl::span<uint8_t>::size_type)> f);
template int forEachRDH(gsl::span<const std::byte> buffer, std::function<void(const RAWDataHeaderV4&, gsl::span<const std::byte>::size_type)> f);

} // namespace o2::mch::raw
