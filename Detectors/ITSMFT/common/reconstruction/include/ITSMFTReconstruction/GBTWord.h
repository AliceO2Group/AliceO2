// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ITSMFT_GBTWORD_H
#define ALICEO2_ITSMFT_GBTWORD_H

// \file GBTWord.h
// \brief Classes for creation/interpretation of ITS/MFT GBT data

#include <Rtypes.h>
#include <cstdint>

namespace o2
{
namespace itsmft
{

constexpr uint64_t LANESMask = (0x1 << 28) - 1; // at most 28 lanes

/// GBT payload header flag
constexpr uint8_t GBTFlagDataHeader = 0xe0;
/// GBT payload trailer flag
constexpr uint8_t GBTFlagDataTrailer = 0xf0;

// GBT header flag in the RDH
constexpr uint8_t GBTFlagRDH = 0x00;
// GBT header flag for the ITS IB: 001 bbbbb with bbbbb -> Lane Number (0-8)
constexpr uint8_t GBTFlagDataIB = 0x20;
// GBT header flag for the ITS OB: 010 bb ccc with bb -> Connector Number (00,01,10,11), ccc -> Lane Number (0-6)
constexpr uint8_t GBTFlagDataOB = 0x40;

constexpr int GBTWordLength = 10;       // lentgh in bytes
constexpr int GBTPaddedWordLength = 16; // lentgh in bytes with padding

class GBTWord
{
  /// GBT word of 80 bits, bits 72:79 are reserver for GBT Header flag, the rest depends on specifications
 public:
  GBTWord() = default;
  GBTWord(uint64_t w0, uint64_t w1) : mData64{w0, w1} {}

  /// check if the GBT Header corresponds to GBT payload header
  static bool isDataHeader(const uint8_t* raw) { return raw[9] == GBTFlagDataHeader; }

  /// check if the GBT Header corresponds to GBT payload trailer
  static bool isDataTrailer(const uint8_t* raw) { return raw[9] == GBTFlagDataTrailer; }

  /// check if the GBT Header corresponds to ITS IB data (header is combined with lanes info)
  static bool isDataIB(const uint8_t* raw) { return (raw[9] & 0xe0) == GBTFlagDataIB; }

  /// check if the GBT Header corresponds to ITS OB data (header is combined with lanes/connector info)
  static bool isDataOB(const uint8_t* raw) { return (raw[9] & 0xe0) == GBTFlagDataOB; }

  /// check if the GBT Header corresponds to GBT payload header
  bool isDataHeader() const { return getHeaderByte() == GBTFlagDataHeader; }

  /// check if the GBT Header corresponds to GBT payload trailer
  bool isDataTrailer() const { return getHeaderByte() == GBTFlagDataTrailer; }

  /// check if the GBT Header corresponds to ITS IB data (header is combined with lanes info)
  bool isDataIB() const { return (getHeaderByte() & 0xe0) == GBTFlagDataIB; }

  /// check if the GBT Header corresponds to ITS OB data (header is combined with lanes/connector info)
  bool isDataOB() const { return (getHeaderByte() & 0xe0) == GBTFlagDataOB; }

  const uint64_t* getW64() const { return mData64; }
  const uint8_t* getW8() const { return mData8; }

  void setByte(uint8_t v, int which) { mData8[which] = v; }
  uint8_t getByte(int which) const { return mData8[which]; }

  uint8_t getHeaderByte() const { return getByte(9); }

  void printX(bool padded = true) const;
  void printB(bool padded = true) const;

 protected:
  union {
    uint8_t mData8[16]; // 80 bits GBT word + optional padding to 128 bits
    uint64_t mData64[2] = {0};
  };
  ClassDefNV(GBTWord, 1);
};

class GBTDataHeader : public GBTWord
{
  /// Definition of ITS/MFT GBT Header: 80 bits long word
  /// In CRU data it must be the 1st word of the payload
  ///
  /// bits  0 : 15, Index of GBT packet within trigger
  /// bits 16 : 43, Active lanes pattern
  /// bits 44 : 71, not used
  /// bits 72 : 79, header/trailer indicator
 public:
  GBTDataHeader() : GBTWord(0, GBTFlagDataHeader << 8) {}
  GBTDataHeader(int packetID, int lanes) : GBTWord(((lanes & LANESMask) << 16) | (packetID & 0xffff), GBTFlagDataHeader << 8) {}

  uint32_t getPacketID() const { return mData64[0] & 0xffff; }
  uint32_t getLanes() const { return (mData64[0] >> 16) & LANESMask; }

  void setPacketID(int id)
  {
    const uint64_t mask = 0xffffffffffff0000;
    mData64[0] &= mask;
    mData64[0] |= (id & 0xffff);
  }
  void setLanes(int lanes)
  {
    constexpr uint64_t mask = ~(LANESMask << 16);
    mData64[0] &= mask;
    mData64[0] |= (lanes & LANESMask) << 16;
  }

  void setByte(uint8_t v, int which) = delete;

  ClassDefNV(GBTDataHeader, 1);
};

class GBTDataTrailer : public GBTWord
{
  /// Definition of ITS/MFT GBT trailer: 80 bits long word
  /// In CRU data it must be the last word of the payload
  ///
  /// bits  0 : 27, Lanes stops received
  /// bits 28 : 31, not used
  /// bits 32 : 59, Lane timeouts received
  /// bits 60 : 63, not used
  /// bits 64 : 71, State of GBT_Packet:
  ///               4: lane_timeouts,  if at least 1 lane timed out
  ///               3: lane_starts_violation,  if at least 1 lane had a start violation
  ///               2: packet_overflow, if max number of packets reached
  ///               1: transmission_timeout, if timeout of transmission (lanes)
  ///               0: packet_done, if Packet finished
 public:
  enum PacketStates : int {
    PacketDone = 0,                              // Packet finished
    TransmissionTimeout = 1,                     // if timeout of transmission (lanes)
    PacketOverflow = 2,                          // if max number of packets reached
    LaneStartsViolation = 3,                     // if at least 1 lane had a start violation
    LameTimeouts = 4,                            // if at least 1 lane timed out
    NStatesDefined,                              // total number of states defined
    MaxStateCombinations = 0x1 << NStatesDefined // number of possible combinations
  };

  GBTDataTrailer() : GBTWord(0, GBTFlagDataTrailer << 8) {}
  GBTDataTrailer(int lanesStop, int lanesTimeOut, int packetState) : GBTWord((lanesStop & LANESMask) | ((lanesTimeOut & LANESMask) << 32),
                                                                             (packetState & 0xff) | (GBTFlagDataTrailer << 8)) {}

  uint32_t getLanesStop() const { return mData64[0] & LANESMask; }
  uint32_t getLanesTimeout() const { return (mData64[0] >> 32) & LANESMask; }
  uint8_t getPacketState() const { return mData8[8]; }

  void setLanesStop(uint32_t lanes)
  {
    constexpr uint64_t mask = ~LANESMask;
    mData64[0] &= mask;
    mData64[0] |= (lanes & LANESMask);
  }

  void setLanesTimeout(uint32_t lanes)
  {
    constexpr uint64_t mask = ~(LANESMask << 32);
    mData64[0] &= mask;
    mData64[0] |= (lanes & LANESMask) << 32;
  }

  void setPacketState(uint8_t v) { mData8[8] = v; }

  void setByte(uint8_t v, int which) = delete;

  ClassDefNV(GBTDataTrailer, 1);
};

class GBTData : public GBTWord
{
  /// Definition of ITS/MFT GBT paiload: 80 bits long word (can be padded to 128b) shipping
  /// Alpide data and having GBTHeader at positions 72:79
 public:
  /// extract lane of the IB as lowest 5 bits (ATTENTION: no check if this is really an IB header)
  static int getLaneIB(uint8_t v) { return v & 0x1f; }
  /// extract connector/lane of the OB as lowest 2/3 bits (ATTENTION: no check if this is really an OB header)
  static int getConnectorOB(uint8_t v) { return (v & 0x18) >> 3; } // lane only
  static int getLaneOB(uint8_t v) { return v & 0x7; }              // connector only
  static int getLaneOB(uint8_t v, int& connector)
  { // separately lane and connector
    connector = getConnectorOB(v);
    return getLaneOB(v);
  }
  static int getCableID(uint8_t v) { return v & 0x1f; } // combined connector and lane

  /// extract lane of the IB as lowest 5 bits (ATTENTION: no check if this is really an IB header)
  int getLaneIB() const { return getLaneIB(getHeaderByte()); }
  /// extract connector/lane of the OB as lowest 2/3 bits (ATTENTION: no check if this is really an OB header)
  int getConnectorOB() const { return getConnectorOB(getHeaderByte()); }                // lane only
  int getLaneOB() const { return getLaneOB(getHeaderByte()); }                          // connector only
  int getLaneOB(int& connector) const { return getLaneOB(getHeaderByte(), connector); } // separately lane and connector
  int getCableID() const { return getCableID(getHeaderByte()); }                        // combined connector and lane

  ClassDefNV(GBTData, 1);
};
} // namespace itsmft
} // namespace o2

#endif
