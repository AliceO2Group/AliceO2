// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawCommon/RDHManip.h"
#include <fmt/format.h>
#include "Headers/RAWDataHeader.h"
#include <iostream>
#include <cassert>

using namespace o2::header;

std::ostream& operator<<(std::ostream& os, const RAWDataHeaderV4& rdh)
{
  os << fmt::format("version              {:03d} headerSize      {:03d} \n",
                    rdh.version,
                    rdh.headerSize);

  os << fmt::format("cruId                {:03d} dpwId            {:02d} linkId        {:03d}\n", rdh.cruID, rdh.endPointID, rdh.linkID);

  os << fmt::format("offsetToNext   {:05d} memorySize    {:05d} blockLength {:05d}\n", rdh.offsetToNext, rdh.memorySize, rdh.blockLength);

  os << fmt::format("triggerOrbit  {:010d} HB orbit {:010d}\n",
                    rdh.triggerOrbit, rdh.heartbeatOrbit);

  os << fmt::format("triggerBC           {:04d} heartbeatBC    {:04d}\n",
                    rdh.triggerBC, rdh.heartbeatBC);

  os << fmt::format("stopBit                {:1d} pagesCounter    {:03d} packetCounter {:03d} \n",
                    rdh.stop, rdh.pageCnt, rdh.packetCounter);

  return os;
}

namespace o2
{
namespace mch
{
namespace raw
{

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

template <>
void appendRDH(std::vector<uint32_t>& buffer, const RAWDataHeaderV4& rdh)
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

template <>
void appendRDH(std::vector<uint8_t>& buffer, const RAWDataHeaderV4& rdh)
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

uint64_t eightBytes(gsl::span<uint8_t> buffer)
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
RAWDataHeaderV4 createRDH(gsl::span<uint8_t> buffer)
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

uint64_t from32(gsl::span<uint32_t> buffer)
{
  return static_cast<uint64_t>(buffer[0]) |
         (static_cast<uint64_t>(buffer[1]) << 32);
}

template <>
RAWDataHeaderV4 createRDH(gsl::span<uint32_t> buffer)
{
  if (buffer.size() < 16) {
    throw std::invalid_argument("buffer should be at least 16 words");
  }
  RAWDataHeaderV4 rdh;
  rdh.word0 = from32(buffer.subspan(0, 2));
  rdh.word1 = from32(buffer.subspan(2, 2));
  rdh.word2 = from32(buffer.subspan(4, 2));
  rdh.word3 = from32(buffer.subspan(6, 2));
  rdh.word4 = from32(buffer.subspan(8, 2));
  rdh.word5 = from32(buffer.subspan(10, 2));
  rdh.word6 = from32(buffer.subspan(12, 2));
  rdh.word7 = from32(buffer.subspan(14, 2));
  return rdh;
}

template <>
RAWDataHeaderV4 createRDH(uint16_t cruId, uint8_t linkId, uint16_t solarId, uint32_t orbit, uint16_t bunchCrossing,
                          uint16_t payloadSize)
{
  RAWDataHeaderV4 rdh;

  if (payloadSize > 0x0FFFF - sizeof(rdh)) {
    throw std::invalid_argument("payload too big");
  }

  uint16_t memorySize = payloadSize + sizeof(rdh);

  rdh.cruID = cruId;
  rdh.linkID = linkId;
  // (need cru mappper : from (cruid,linkid)->(solarid) ? and then we pass cruId,linkId to this function
  // and deduce solarId instead ?)
  rdh.endPointID = 0;  // FIXME: fill this ?
  rdh.feeId = solarId; //FIXME: what is this field supposed to contain ? unclear to me.
  rdh.priority = 0;
  rdh.blockLength = memorySize - sizeof(rdh); // FIXME: the blockLength disappears in RDHv5 ?
  rdh.memorySize = memorySize;
  rdh.offsetToNext = memorySize;
  rdh.packetCounter = 0; // FIXME: fill this ?
  rdh.triggerType = 0;   // FIXME: fill this ?
  rdh.detectorField = 0; // FIXME: fill this ?
  rdh.par = 0;           // FIXME: fill this ?
  rdh.stop = 0;
  rdh.pageCnt = 1;
  rdh.triggerOrbit = orbit;
  rdh.heartbeatOrbit = orbit; // FIXME: RDHv5 has only triggerOrbit ?
  rdh.triggerBC = bunchCrossing;
  rdh.heartbeatBC = bunchCrossing; // FIXME: RDHv5 has only triggerBC ?

  return rdh;
}

template <>
uint32_t rdhOrbit(const RAWDataHeaderV4& rdh)
{
  return rdh.triggerOrbit; // or is it heartbeatOrbit ?
}

template <>
size_t rdhPayloadSize(const RAWDataHeaderV4& rdh)
{
  return rdh.memorySize - sizeof(rdh);
}

template <>
uint8_t rdhLinkId(const RAWDataHeaderV4& rdh)
{
  return rdh.linkID + 12 * rdh.endPointID;
}

template <>
uint16_t rdhBunchCrossing(const RAWDataHeaderV4& rdh)
{
  return static_cast<uint16_t>(rdh.triggerBC & 0xFFF);
}

void dumpRDHBuffer(gsl::span<uint32_t> buffer, std::string_view indent)
{
  auto const rdh = createRDH<o2::header::RAWDataHeaderV4>(buffer);
  std::cout << fmt::format("{:016X} {:016X}",
                           rdh.word1, rdh.word0);
  std::cout << fmt::format(" version {:d} headerSize {:d} blockLength {:d} \n",
                           rdh.version, rdh.headerSize, rdh.blockLength);
  std::cout << fmt::format("{:44s} feeId {} priority {}\n", " ", rdh.feeId, rdh.priority);
  std::cout << fmt::format("{:44s} offsetnext {} memsize {}\n", " ", rdh.offsetToNext, rdh.memorySize);
  std::cout << fmt::format("{:44s} linkId {} packetCount {} cruId {} dpwId {}\n", " ", rdh.linkID, rdh.packetCounter, rdh.cruID, rdh.endPointID);

  std::cout << indent << fmt::format("{:016X} {:016X}", rdh.word3, rdh.word2);
  std::cout << fmt::format(" triggerOrbit {:d} \n", rdh.triggerOrbit);
  std::cout << fmt::format("{:44s} heartbeatOrbit {}\n", " ", rdh.heartbeatOrbit);
  std::cout << fmt::format("{:44s} zero\n", " ");
  std::cout << fmt::format("{:44s} zero\n", " ");

  std::cout << indent << fmt::format("{:016X} {:016X}", rdh.word5, rdh.word4);
  std::cout << fmt::format(" triggerBC {}  heartbeatBC {}\n", rdh.triggerBC,
                           rdh.heartbeatBC);
  std::cout << fmt::format("{:44s} triggerType {}\n", " ", rdh.triggerType);
  std::cout << fmt::format("{:44s} zero\n", " ");
  std::cout << fmt::format("{:44s} zero\n", " ");

  std::cout << indent << fmt::format("{:016X} {:016X}", rdh.word7, rdh.word6);
  std::cout << fmt::format(" detectorField {}  par {}\n", rdh.detectorField,
                           rdh.par);
  std::cout << fmt::format("{:44s} stopBit {} pagesCounter {}\n", " ",
                           rdh.stop, rdh.pageCnt);
  std::cout << fmt::format("{:44s} zero\n", " ");
  std::cout << fmt::format("{:44s} zero\n", " ");
}

template <typename RDH>
int forEachRDH(gsl::span<uint32_t> buffer, std::function<void(RDH&)> f)
{
  RDH rdh;
  int index{0};
  int nrdh{0};
  while (index < buffer.size()) {
    memcpy(&rdh, &buffer[0] + index, sizeof(rdh));
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
    index += rdh.offsetToNext / 4;
  }
  return nrdh;
}

template <typename RDH>
int showRDHs(gsl::span<uint32_t> buffer)
{
  return forEachRDH<RDH>(buffer, [](RDH& rdh) {
    std::cout << rdh << "\n";
  });
}

template <typename RDH>
int showRDHs(gsl::span<uint8_t> buffer)
{
  assert(buffer.size() % 4 == 0);
  gsl::span<uint32_t> buf32{reinterpret_cast<uint32_t*>(&buffer[0]),
                            buffer.size() / 4};
  return showRDHs<RDH>(buf32);
}

template <typename RDH>
int countRDHs(gsl::span<uint32_t> buffer)
{
  return forEachRDH<RDH>(buffer, nullptr);
}

template <typename RDH>
int countRDHs(gsl::span<uint8_t> buffer)
{
  assert(buffer.size() % 4 == 0);
  gsl::span<uint32_t> buf32{reinterpret_cast<uint32_t*>(&buffer[0]),
                            buffer.size() / 4};
  return countRDHs<RDH>(buf32);
}

// force instanciation of (only) the templates we need

// RDH v4 versions

template int showRDHs<RAWDataHeaderV4>(gsl::span<uint32_t> buffer);
template int showRDHs<RAWDataHeaderV4>(gsl::span<uint8_t> buffer);
template int countRDHs<RAWDataHeaderV4>(gsl::span<uint8_t> buffer);
template int countRDHs<RAWDataHeaderV4>(gsl::span<uint32_t> buffer);
template int forEachRDH(gsl::span<uint32_t> buffer, std::function<void(RAWDataHeaderV4&)> f);

} // namespace raw
} // namespace mch
} // namespace o2
