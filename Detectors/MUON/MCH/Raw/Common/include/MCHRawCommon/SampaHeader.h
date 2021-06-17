// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_SAMPA_HEADER
#define O2_MCH_RAW_SAMPA_HEADER

#include <cstdlib>
#include <iostream>
#include "MCHRawCommon/DataFormats.h"

namespace o2
{
namespace mch
{
namespace raw
{

enum class SampaPacketType : uint8_t {
  HeartBeat = 0,
  DataTruncated = 1,
  Sync = 2,
  DataTruncatedTriggerTooEarly = 3,
  Data = 4,
  DataNumWords = 5,
  DataTriggerTooEarly = 6,
  DataTriggerTooEarlyNumWords = 7
};

/// @brief SampaHeader is the 50-bits header word used in Sampa data transmission protocol.
///
/// - hamming[0..5] (6 bits) is the Hamming code of the header itself
/// - p[6] (1 bit) is the parity (odd) of the header including hamming
/// - pkt[7..9] (3 bits) is the packet type
/// - numWords[10..19] (10 bits) is the number of 10 bit words in the data payload
/// - h[20..23] (4 bits) is the hardware address of the chip
/// - ch[24..28] (5 bits) is the channel address
/// - bx[29..48] (20 bits) is the bunch-crossing counter (40 MHz counter)
/// - dp[49] (1 bit) is the parity (odd) of data payload
///
/// \nosubgrouping

class SampaHeader
{
 public:
  explicit SampaHeader(uint64_t value = 0);

  /// Constructor.
  /// \param hamming 6 bits
  /// \param p 1 bit
  /// \param pkt 3 bits
  /// \param numWords 10 bits
  /// \param h 4 bits
  /// \param ch 5 bits
  /// \param bx 20 bits
  /// \param dp 1 bit
  ///
  /// if any of the parameter is not in its expected range,
  /// the ctor throws an exception
  explicit SampaHeader(uint8_t hamming,
                       bool p,
                       SampaPacketType pkt,
                       uint16_t numWords,
                       uint8_t h,
                       SampaChannelAddress ch,
                       uint32_t bx,
                       bool dp);

  bool operator==(const SampaHeader& rhs) const;
  bool operator!=(const SampaHeader& rhs) const;
  bool operator<(const SampaHeader& rhs) const;
  bool operator<=(const SampaHeader& rhs) const;
  bool operator>(const SampaHeader& rhs) const;
  bool operator>=(const SampaHeader& rhs) const;

  /// whether the header has error (according to hamming and/or parity)
  bool hasError() const;
  bool hasHammingError() const;
  bool hasParityError() const;

  /** @name Getters
    */
  ///@{
  uint8_t hammingCode() const;
  bool headerParity() const;
  SampaPacketType packetType() const;
  uint16_t nof10BitWords() const;
  uint8_t chipAddress() const;
  SampaChannelAddress channelAddress() const;
  uint32_t bunchCrossingCounter() const;
  bool payloadParity() const;
  ///@}

  /** @name Setters
    Each setter throws if the value does not fit in the expected number of bits.
    */
  ///@{
  void hammingCode(uint8_t hamming);
  void headerParity(bool p);
  void packetType(SampaPacketType pkt);
  void nof10BitWords(uint16_t nofwords);
  void chipAddress(uint8_t h);
  void channelAddress(SampaChannelAddress ch);
  void bunchCrossingCounter(uint32_t bx);
  void payloadParity(bool dp);
  ///@}

  /// return the header as a 64-bits integer
  constexpr uint64_t uint64() const { return mValue; }

  /// sets the value from a 64-bits integer.
  /// if the value does not fit within 50 bits an exception is thrown.
  void uint64(uint64_t value);

  bool isHeartbeat() const;

 private:
  uint64_t mValue;
};

constexpr uint64_t sampaSyncWord{0x1555540f00113};

/// Whether the 50 LSB bits match the sync word
constexpr bool isSampaSync(uint64_t w)
{
  constexpr uint64_t FIFTYBITSATONE = (static_cast<uint64_t>(1) << 50) - 1;
  return ((w & FIFTYBITSATONE) == sampaSyncWord);
}

/// The 50-bits Sampa SYNC word.
SampaHeader sampaSync();

/// Heartbeat packet
SampaHeader sampaHeartbeat(uint8_t elinkId, uint20_t bunchCrossing);

/// Return channel number (0..63)
DualSampaChannelId getDualSampaChannelId(const SampaHeader& sh);

/// packetTypeName returns a string representation of the given packet type.
std::string packetTypeName(SampaPacketType pkt);

/// compute the hamming code of value
/// assuming it is 50 bits
/// and represents a Sampa header
int computeHammingCode(uint64_t value);

int computeHammingCode1(uint64_t value);
int computeHammingCode2(uint64_t value);
int computeHammingCode3(uint64_t value);
int computeHammingCode4(uint64_t value);

/// compute parity of value
/// assuming it is 50 bits
/// and represents a Sampa header
/// (not using the existing header parity at bit pos 6)
int computeHeaderParity(uint64_t value);

int computeHeaderParity1(uint64_t value);
int computeHeaderParity2(uint64_t value);
int computeHeaderParity3(uint64_t value);
int computeHeaderParity4(uint64_t value);

std::string asString(const SampaHeader& sh);

std::ostream&
  operator<<(std::ostream& os, const SampaHeader& sh);

} // namespace raw
} // namespace mch
} // namespace o2

#endif
