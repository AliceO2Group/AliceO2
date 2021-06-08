// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawCommon/SampaHeader.h"

#include <stdexcept>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "NofBits.h"
#include <array>
#include <sstream>

namespace
{
constexpr uint64_t HAMMING_CODE_OFFSET = 0;
constexpr uint64_t HEADER_PARITY_OFFSET = 6;
constexpr uint64_t PACKET_TYPE_OFFSET = 7;
constexpr uint64_t NUMBER_OF_1OBITS_WORDS_OFFSET = 10;
constexpr uint64_t CHIP_ADDRESS_OFFSET = 20;
constexpr uint64_t CHANNEL_ADDRESS_OFFSET = 24;
constexpr uint64_t BUNCH_CROSSING_OFFSET = 29;
constexpr uint64_t PARITY_OFFSET = 49;

constexpr uint64_t HAMMING_CODE_NOFBITS = HEADER_PARITY_OFFSET - HAMMING_CODE_OFFSET;
constexpr uint64_t HEADER_PARITY_NOFBITS = PACKET_TYPE_OFFSET - HEADER_PARITY_OFFSET;
constexpr uint64_t PACKET_TYPE_NOFBITS = NUMBER_OF_1OBITS_WORDS_OFFSET - PACKET_TYPE_OFFSET;
constexpr uint64_t NUMBER_OF_1OBITS_WORDS_NOFBITS = CHIP_ADDRESS_OFFSET - NUMBER_OF_1OBITS_WORDS_OFFSET;
constexpr uint64_t CHIP_ADDRESS_NOFBITS = CHANNEL_ADDRESS_OFFSET - CHIP_ADDRESS_OFFSET;
constexpr uint64_t CHANNEL_ADDRESS_NOFBITS = BUNCH_CROSSING_OFFSET - CHANNEL_ADDRESS_OFFSET;
constexpr uint64_t BUNCH_CROSSING_NOFBITS = PARITY_OFFSET - BUNCH_CROSSING_OFFSET;
constexpr uint64_t PARITY_NOFBITS = 50 - PARITY_OFFSET;

struct CHECKNOFBITS {

  CHECKNOFBITS()
  {
    if (HAMMING_CODE_NOFBITS != 6) {
      throw std::invalid_argument(fmt::format("HAMMING_CODE_NOFBITS is {0}. Should be 6", HAMMING_CODE_NOFBITS));
    }
    if (HEADER_PARITY_NOFBITS != 1) {
      throw std::invalid_argument(fmt::format("HEADER_PARITY_NOFBITS is {0}. Should be 1", HEADER_PARITY_NOFBITS));
    }
    if (PACKET_TYPE_NOFBITS != 3) {
      throw std::invalid_argument(fmt::format("PACKET_TYPE_NOFBITS is {0}. Should be 3", PACKET_TYPE_NOFBITS));
    }
    if (NUMBER_OF_1OBITS_WORDS_NOFBITS != 10) {
      throw std::invalid_argument(fmt::format("NUMBER_OF_1OBITS_WORDS_NOFBITS is {0}. Should be 10", NUMBER_OF_1OBITS_WORDS_NOFBITS));
    }
    if (CHIP_ADDRESS_NOFBITS != 4) {
      throw std::invalid_argument(fmt::format("CHIP_ADDRESS_NOFBITS is {0}. Should be 4", CHIP_ADDRESS_NOFBITS));
    }
    if (CHANNEL_ADDRESS_NOFBITS != 5) {
      throw std::invalid_argument(fmt::format("CHANNEL_ADDRESS_NOFBITS is {0}. Should be 5", CHANNEL_ADDRESS_NOFBITS));
    }
    if (BUNCH_CROSSING_NOFBITS != 20) {
      throw std::invalid_argument(fmt::format("BUNCH_CROSSING_NOFBITS is {0}. Should be 20", BUNCH_CROSSING_NOFBITS));
    }
    if (PARITY_NOFBITS != 1) {
      throw std::invalid_argument(fmt::format("PARITY_NOFBITS is {0}. Should be 1", PARITY_NOFBITS));

      throw;
    }
  }
};

CHECKNOFBITS checknofbits;

constexpr uint64_t HAMMING_CODE_MASK = 0x000000000003F;
constexpr uint64_t HEADER_PARITY_MASK = 0x0000000000040;
constexpr uint64_t PACKET_TYPE_MASK = 0x0000000000380;
constexpr uint64_t NUMBER_OF_1OBITS_WORDS_MASK = 0x00000000FFC00;
constexpr uint64_t CHIP_ADDRESS_MASK = 0x0000000F00000;
constexpr uint64_t CHANNEL_ADDRESS_MASK = 0x000001F000000;
constexpr uint64_t BUNCH_CROSSING_MASK = 0x1FFFFE0000000;
constexpr uint64_t PARITY_MASK = 0x2000000000000;

constexpr uint64_t hammingCode(uint64_t header)
{
  return ((header & HAMMING_CODE_MASK) >> HAMMING_CODE_OFFSET);
}

constexpr uint64_t headerParity(uint64_t header)
{
  return ((header & HEADER_PARITY_MASK) >> HEADER_PARITY_OFFSET);
}
constexpr uint64_t packetType(uint64_t header)
{
  return ((header & PACKET_TYPE_MASK) >> PACKET_TYPE_OFFSET);
}

constexpr uint64_t nof10BitWords(uint64_t header)
{
  return ((header & NUMBER_OF_1OBITS_WORDS_MASK) >> NUMBER_OF_1OBITS_WORDS_OFFSET);
}

constexpr uint64_t chipAddress(uint64_t header)
{
  return ((header & CHIP_ADDRESS_MASK) >> CHIP_ADDRESS_OFFSET);
}
constexpr uint64_t channelAddress(uint64_t header)
{
  return ((header & CHANNEL_ADDRESS_MASK) >> CHANNEL_ADDRESS_OFFSET);
}

constexpr uint64_t bunchCrossingCounter(uint64_t header)
{
  return ((header & BUNCH_CROSSING_MASK) >> BUNCH_CROSSING_OFFSET);
}

constexpr uint64_t payloadParity(uint64_t header)
{
  return ((header & PARITY_MASK) >> PARITY_OFFSET);
}

bool checkBit(uint64_t value, int i, bool bitStatus)
{
  return (value >> i & 1) == static_cast<int>(bitStatus);
}

bool isHeartbeat(uint64_t header)
{
  // - bits 7-9 must be zero
  // - bits 10-19 must be zero
  // - bits 24,26,28 must be one
  // - bits 25,27 must be zero
  // - bit 49 must be zero
  for (int i = 7; i <= 9; i++) {
    if (!checkBit(header, i, false)) {
      return false;
    }
  }
  for (int i = 10; i <= 19; i++) {
    if (!checkBit(header, i, false)) {
      return false;
    }
  }
  if (!checkBit(header, 24, true) || !checkBit(header, 26, true) || !checkBit(header, 28, true) || !checkBit(header, 25, false) || !checkBit(header, 27, false) || !checkBit(header, 49, false)) {
    return false;
  }
  return true;
}

} // namespace

namespace o2
{
namespace mch
{
namespace raw
{

std::string packetTypeName(o2::mch::raw::SampaPacketType pkt)
{
  if (pkt == o2::mch::raw::SampaPacketType::HeartBeat) {
    return "HeartBeat";
  }
  if (pkt == o2::mch::raw::SampaPacketType::DataTruncated) {
    return "DataTruncated";
  }
  if (pkt == o2::mch::raw::SampaPacketType::Sync) {
    return "Sync";
  }
  if (pkt == o2::mch::raw::SampaPacketType::DataTruncatedTriggerTooEarly) {
    return "DataTruncatedTriggerTooEarly";
  }
  if (pkt == o2::mch::raw::SampaPacketType::Data) {
    return "Data";
  }
  if (pkt == o2::mch::raw::SampaPacketType::DataNumWords) {
    return "DataNumWords";
  }
  if (pkt == o2::mch::raw::SampaPacketType::DataTriggerTooEarly) {
    return "DataTriggerTooEarl";
  }
  if (pkt == o2::mch::raw::SampaPacketType::DataTriggerTooEarlyNumWords) {
    return "DataTriggerTooEarlyNumWords";
  }
  throw std::out_of_range("should not happen");
}

SampaHeader::SampaHeader(uint64_t value) : mValue(0)
{
  uint64(value);
}

bool SampaHeader::hasHammingError() const
{
  return computeHammingCode(mValue) != hammingCode();
}

bool SampaHeader::hasError() const
{
  return hasHammingError() || hasParityError();
}

bool SampaHeader::hasParityError() const
{
  return computeHeaderParity(mValue) != headerParity();
}

void SampaHeader::uint64(uint64_t value)
{
  impl::assertNofBits("sampa header", value, 50);
  mValue = value;
}

SampaHeader::SampaHeader(uint8_t hamming,
                         bool p,
                         SampaPacketType pkt,
                         uint16_t numWords,
                         uint8_t h,
                         uint8_t ch,
                         uint32_t bx,
                         bool dp) : mValue{0}
{
  hammingCode(hamming);
  headerParity(p);
  packetType(pkt);
  nof10BitWords(numWords);
  chipAddress(h);
  channelAddress(ch);
  bunchCrossingCounter(bx);
  payloadParity(dp);
}

void SampaHeader::headerParity(bool p)
{
  mValue &= ~HEADER_PARITY_MASK;
  mValue += (static_cast<uint64_t>(p) << HEADER_PARITY_OFFSET) & HEADER_PARITY_MASK;
}

void SampaHeader::payloadParity(bool dp)
{
  mValue &= ~PARITY_MASK;
  mValue += (static_cast<uint64_t>(dp) << PARITY_OFFSET) & PARITY_MASK;
}

void SampaHeader::chipAddress(uint8_t h)
{
  mValue &= ~CHIP_ADDRESS_MASK;
  mValue += (static_cast<uint64_t>(h) << CHIP_ADDRESS_OFFSET) & CHIP_ADDRESS_MASK;
}

void SampaHeader::channelAddress(uint8_t ch)
{
  mValue &= ~CHANNEL_ADDRESS_MASK;
  mValue += (static_cast<uint64_t>(ch) << CHANNEL_ADDRESS_OFFSET) & CHANNEL_ADDRESS_MASK;
}

void SampaHeader::bunchCrossingCounter(uint32_t bx)
{
  mValue &= ~BUNCH_CROSSING_MASK;
  mValue += (static_cast<uint64_t>(bx) << BUNCH_CROSSING_OFFSET) & BUNCH_CROSSING_MASK;
}

void SampaHeader::hammingCode(uint8_t hamming)
{
  mValue &= ~HAMMING_CODE_MASK;
  mValue += (static_cast<uint64_t>(hamming) << HAMMING_CODE_OFFSET) & HAMMING_CODE_MASK;
}

void SampaHeader::nof10BitWords(uint16_t nofwords)
{
  mValue &= ~NUMBER_OF_1OBITS_WORDS_MASK;
  mValue += (static_cast<uint64_t>(nofwords) << NUMBER_OF_1OBITS_WORDS_OFFSET) & NUMBER_OF_1OBITS_WORDS_MASK;
}

void SampaHeader::packetType(SampaPacketType pkt)
{
  mValue &= ~PACKET_TYPE_MASK;
  mValue += (static_cast<uint64_t>(pkt) << PACKET_TYPE_OFFSET) & PACKET_TYPE_MASK;
}

bool SampaHeader::operator<(const SampaHeader& rhs) const
{
  return bunchCrossingCounter() < rhs.bunchCrossingCounter();
}

bool SampaHeader::operator>(const SampaHeader& rhs) const
{
  return bunchCrossingCounter() > rhs.bunchCrossingCounter();
}

bool SampaHeader::operator<=(const SampaHeader& rhs) const
{
  return bunchCrossingCounter() <= rhs.bunchCrossingCounter();
}

bool SampaHeader::operator>=(const SampaHeader& rhs) const
{
  return bunchCrossingCounter() >= rhs.bunchCrossingCounter();
}

bool SampaHeader::operator==(const SampaHeader& rhs) const
{
  return mValue == rhs.mValue;
}

bool SampaHeader::operator!=(const SampaHeader& rhs) const
{
  return !(*this == rhs);
}

bool SampaHeader::isHeartbeat() const
{
  return ::isHeartbeat(mValue);
}
uint8_t SampaHeader::hammingCode() const
{
  // 6 bits
  return ::hammingCode(mValue) & 0X3F;
}

bool SampaHeader::headerParity() const
{
  return ::headerParity(mValue) == 1;
}

SampaPacketType SampaHeader::packetType() const
{
  // 3 bits
  return static_cast<SampaPacketType>(::packetType(mValue) & 0x7);
}

uint16_t SampaHeader::nof10BitWords() const
{
  // 10 bits
  return ::nof10BitWords(mValue) & 0x3FF;
}

uint8_t SampaHeader::chipAddress() const
{
  // 4 bits
  return ::chipAddress(mValue) & 0xF;
}

uint8_t SampaHeader::channelAddress() const
{
  // 5 bits
  return ::channelAddress(mValue) & 0x1F;
}

uint32_t SampaHeader::bunchCrossingCounter() const
{
  // 20 bits
  return ::bunchCrossingCounter(mValue) & 0x1FFFFF;
}

bool SampaHeader::payloadParity() const
{
  return ::payloadParity(mValue) == 1;
}

SampaHeader sampaSync()
{
  return SampaHeader(
    0x13,
    0,
    SampaPacketType::Sync,
    0,
    0xf,
    0,
    0xAAAAA,
    0);
}

SampaHeader sampaHeartbeat(uint8_t chipAddress, uint20_t bunchCrossing)
{
  SampaHeader sh;
  sh.packetType(SampaPacketType::HeartBeat);
  sh.nof10BitWords(0);
  sh.chipAddress(chipAddress);
  sh.channelAddress(21);
  sh.bunchCrossingCounter(bunchCrossing);
  sh.hammingCode(computeHammingCode(sh.uint64()));
  sh.headerParity(computeHeaderParity(sh.uint64()));
  return sh;
}

std::string asString(const SampaHeader& sh)
{
  std::stringstream os;

  uint64_t one = 1;
  std::vector<int> sep = {5, 6, 9, 19, 23, 28, 48};
  for (int i = 0; i < 50; i++) {
    os << ((sh.uint64() & (one << i)) ? "1" : "0");
    if (std::find(begin(sep), end(sep), i) != end(sep)) {
      os << "|";
    }
  }
  os << "\n";
  os << fmt::sprintf("%6x %d %3x %10x %4x %5x %20x %d (0x%x) | HeaderType: %s %s",
                     sh.hammingCode(),
                     sh.headerParity(),
                     static_cast<uint8_t>(sh.packetType()),
                     sh.nof10BitWords(),
                     sh.chipAddress(),
                     sh.channelAddress(),
                     sh.bunchCrossingCounter(),
                     sh.payloadParity(),
                     sh.uint64(),
                     packetTypeName(sh.packetType()),
                     sh.hasHammingError() ? " | HAMMING_ERROR" : "",
                     sh.hasParityError() ? "| PARITY ERROR" : "")
     << "\n";
  return os.str();
}

std::ostream& operator<<(std::ostream& os, const SampaHeader& sh)
{
  return os << asString(sh);
}

// conv array convert a bit position in the hamming sense
// (i.e. where parity bits are interleaved between data bits)
// and the data bit positions in the original value (where the 6 hamming
// bits are "grouped" in the front of the value).
constexpr std::array<int, 49> conv = {-1, -1, 7, -1, 8, 9, 10, -1, 11, 12, 13, 14, 15, 16, 17,
                                      -1, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                                      29, 30, 31, 32, -1, 33, 34, 35, 36, 37, 38, 39,
                                      40, 41, 42, 43, 44, 45, 46, 47, 48, 49};
int computeParity(uint64_t v);

int partialOddParity3(uint64_t value, int pos)
{
  // compute the odd parity of all the bits at position x
  // (where x & (2^pos)) are set
  //

  int n{0};
  uint64_t one{1};
  const uint64_t test{one << pos};

  for (uint64_t i = 0; i < 49; i++) {
    if (conv[i] < 0) {
      continue;
    }
    if ((i + 1) & test) {
      if (value & (one << conv[i])) {
        ++n;
      }
    }
  }
  return (n + 1) % 2 == 0;
}

int computeHammingCode3(uint64_t value)
{
  // value is assumed to be 50 bits, where the 43 data bits
  // are 7-49
  int hamming{0};

  for (int i = 0; i < 6; i++) {
    hamming += partialOddParity3(value, i) * (1 << i);
  }
  return hamming;
}

template <size_t N>
int partialOddParity2(uint64_t value, const std::array<int, N>& pos)
{
  uint64_t one{1};
  int n{0};

  for (auto i = 0; i < pos.size(); i++) {
    n += ((value & (one << pos[i])) > 0);
  }
  return (n + 1) % 2 == 0;
}

template <size_t N>
int partialOddParity(uint64_t value, const std::array<uint64_t, N>& masks)
{
  int n{0};
  for (auto i = 0; i < masks.size(); i++) {
    n += ((value & masks[i]) > 0);
  }
  return (n + 1) % 2 == 0;
}

std::array<int, 24> p0{7, 8, 10, 11, 13, 15, 17, 18, 20, 22, 24, 26, 28, 30, 32, 33, 35, 37, 39, 41, 43, 45, 47, 49};
std::array<int, 23> p1{7, 9, 10, 12, 13, 16, 17, 19, 20, 23, 24, 27, 28, 31, 32, 34, 35, 38, 39, 42, 43, 46, 47};
std::array<int, 23> p2{8, 9, 10, 14, 15, 16, 17, 21, 22, 23, 24, 29, 30, 31, 32, 36, 37, 38, 39, 44, 45, 46, 47};
std::array<int, 23> p3{11, 12, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 30, 31, 32, 40, 41, 42, 43, 44, 45, 46, 47};
std::array<int, 17> p4{18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 48, 49};
std::array<int, 17> p5{33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49};

int computeHammingCode2(uint64_t value)
{
  // value is assumed to be 50 bits, where the 43 data bits
  // are 7-49

  int h0 = partialOddParity2(value, p0);
  int h1 = partialOddParity2(value, p1);
  int h2 = partialOddParity2(value, p2);
  int h3 = partialOddParity2(value, p3);
  int h4 = partialOddParity2(value, p4);
  int h5 = partialOddParity2(value, p5);
  return h0 + h1 * 2 + h2 * 4 + h3 * 8 + h4 * 16 + h5 * 32;
}

std::array<uint64_t, 24> m0 = {0x80, 0x100, 0x400, 0x800, 0x2000, 0x8000, 0x20000, 0x40000, 0x100000, 0x400000, 0x1000000, 0x4000000, 0x10000000, 0x40000000, 0x100000000, 0x200000000, 0x800000000, 0x2000000000, 0x8000000000, 0x20000000000, 0x80000000000, 0x200000000000, 0x800000000000, 0x2000000000000};
std::array<uint64_t, 23> m1 = {0x80, 0x200, 0x400, 0x1000, 0x2000, 0x10000, 0x20000, 0x80000, 0x100000, 0x800000, 0x1000000, 0x8000000, 0x10000000, 0x80000000, 0x100000000, 0x400000000, 0x800000000, 0x4000000000, 0x8000000000, 0x40000000000, 0x80000000000, 0x400000000000, 0x800000000000};
std::array<uint64_t, 23> m2 = {0x100, 0x200, 0x400, 0x4000, 0x8000, 0x10000, 0x20000, 0x200000, 0x400000, 0x800000, 0x1000000, 0x20000000, 0x40000000, 0x80000000, 0x100000000, 0x1000000000, 0x2000000000, 0x4000000000, 0x8000000000, 0x100000000000, 0x200000000000, 0x400000000000, 0x800000000000};
std::array<uint64_t, 23> m3 = {0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000, 0x20000, 0x2000000, 0x4000000, 0x8000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000, 0x100000000, 0x10000000000, 0x20000000000, 0x40000000000, 0x80000000000, 0x100000000000, 0x200000000000, 0x400000000000, 0x800000000000};
std::array<uint64_t, 17> m4 = {0x40000, 0x80000, 0x100000, 0x200000, 0x400000, 0x800000, 0x1000000, 0x2000000, 0x4000000, 0x8000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000, 0x100000000, 0x1000000000000, 0x2000000000000};
std::array<uint64_t, 17> m5 = {0x200000000, 0x400000000, 0x800000000, 0x1000000000, 0x2000000000, 0x4000000000, 0x8000000000, 0x10000000000, 0x20000000000, 0x40000000000, 0x80000000000, 0x100000000000, 0x200000000000, 0x400000000000, 0x800000000000, 0x1000000000000, 0x2000000000000};

int computeHammingCode(uint64_t value)
{
  return computeHammingCode4(value);
}

int computeHammingCode1(uint64_t value)
{
  // value is assumed to be 50 bits, where the 43 data bits
  // are 7-49

  int h0 = partialOddParity(value, m0);
  int h1 = partialOddParity(value, m1);
  int h2 = partialOddParity(value, m2);
  int h3 = partialOddParity(value, m3);
  int h4 = partialOddParity(value, m4);
  int h5 = partialOddParity(value, m5);
  return h0 + h1 * 2 + h2 * 4 + h3 * 8 + h4 * 16 + h5 * 32;
}

template <size_t N>
uint64_t computeMask(const std::array<uint64_t, N>& masks)
{
  uint64_t v{0};
  for (auto i = 0; i < masks.size(); i++) {
    v |= masks[i];
  }
  std::cout << fmt::format("0x{:X},", v);
  return v;
}

int computeHammingCode4(uint64_t value)
{
  // value is assumed to be 50 bits, where the 43 data bits
  // are 7-49

  constexpr std::array<uint64_t, 6> masks = {0x2AAAB5556AD80,
                                             0xCCCD999B3680, 0xF0F1E1E3C700,
                                             0xFF01FE03F800, 0x30001FFFC0000,
                                             0x3FFFE00000000};
  int h0 = computeParity(value & masks[0]);
  int h1 = computeParity(value & masks[1]);
  int h2 = computeParity(value & masks[2]);
  int h3 = computeParity(value & masks[3]);
  int h4 = computeParity(value & masks[4]);
  int h5 = computeParity(value & masks[5]);
  return h0 + h1 * 2 + h2 * 4 + h3 * 8 + h4 * 16 + h5 * 32;
}

int computeHeaderParity(uint64_t v)
{
  return computeHeaderParity4(v);
}

/// compute parity of v, assuming it is 50 bits and
/// represents a Sampa header
/// (not using the existing header parity at bit position 6)
int computeHeaderParity1(uint64_t v)
{
  int n{0};
  constexpr uint64_t one{1};
  for (int i = 0; i < 6; i++) {
    if (v & (one << i)) {
      n++;
    }
  }
  for (int i = 7; i < 50; i++) {
    if (v & (one << i)) {
      n++;
    }
  }
  return (n + 1) % 2 == 0;
}

constexpr std::array<uint64_t, 49> parityMasks = {0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000, 0x20000, 0x40000, 0x80000, 0x100000, 0x200000, 0x400000, 0x800000, 0x1000000, 0x2000000, 0x4000000, 0x8000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000, 0x100000000, 0x200000000, 0x400000000, 0x800000000, 0x1000000000, 0x2000000000, 0x4000000000, 0x8000000000, 0x10000000000, 0x20000000000, 0x40000000000, 0x80000000000, 0x100000000000, 0x200000000000, 0x400000000000, 0x800000000000, 0x1000000000000, 0x2000000000000};

int computeHeaderParity2(uint64_t v)
{
  int n{0};

  for (auto i = 0; i < parityMasks.size(); i++) {
    if (v & parityMasks[i]) {
      n++;
    }
  }
  return (n + 1) % 2 == 0;
}

int computeHeaderParity3(uint64_t no)
{
  int result = 0;
  no &= ~(1UL << 6); // reset bit 6
  while (no != 0) {
    no = no & (no - 1);
    result ^= 1;
  }

  return (short)(result & 0x1);
}

int computeParity(uint64_t no)
{
  no ^= (no >> 32);
  no ^= (no >> 16);
  no ^= (no >> 8);
  no ^= (no >> 4);
  no ^= (no >> 2);
  no ^= (no >> 1);
  return (short)(no & 0x1);
}

int computeHeaderParity4(uint64_t no)
{
  constexpr uint64_t one{1};
  no &= ~(one << 6); // reset bit 6
  return computeParity(no);
}

DualSampaChannelId getDualSampaChannelId(const SampaHeader& sh)
{
  return sh.channelAddress() + (sh.chipAddress() % 2) * 32;
}
} // namespace raw
} // namespace mch
} // namespace o2
