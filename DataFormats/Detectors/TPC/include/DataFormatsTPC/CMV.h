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

/// @file   CMV.h
/// @author Tuba Gündem, tuba.gundem@cern.ch
/// @brief  Common mode values data format definition

/// The data is sent by the CRU as 256+16 bit words. The CMV data layout is as follows:
/// - 256-bit Header: [version:8][packetID:8][errorCode:8][magicWord:8][heartbeatOrbit:32][heartbeatBC:16][padding:176]
/// - 16-bit CMV value: [sign:1][I8F7:15] where bit 15 is the sign (1=positive, 0=negative) and the lower 15 bits are a fixed point I8F7 value (8 integer bits, 7 fractional bits)
///   Float conversion: sign ? (value & 0x7FFF) / 128.0 : -(value & 0x7FFF) / 128.0

#ifndef ALICEO2_DATAFORMATSTPC_CMV_H
#define ALICEO2_DATAFORMATSTPC_CMV_H

#include <cstdint>
#include <cmath>

namespace o2::tpc::cmv
{

static constexpr uint32_t NTimeBinsPerPacket = 3564;                                 ///< number of time bins (covering 8 heartbeats)
static constexpr uint32_t NPacketsPerTFPerCRU = 4;                                   ///< 4 packets per timeframe
static constexpr uint32_t NTimeBinsPerTF = NTimeBinsPerPacket * NPacketsPerTFPerCRU; ///< maximum number of timebins per timeframe (14256)

/// Data padding: NTimeBinsPerPacket * sizeof(Data) = 3564 * 2 = 7128 bytes
static constexpr uint32_t DataSizeBytes = NTimeBinsPerPacket * sizeof(uint16_t); ///< 7128 bytes
static constexpr uint32_t DataPaddingBytes = (32 - (DataSizeBytes % 32)) % 32;   ///< 8 bytes

/// Header definition of the CMVs
struct Header {
  static constexpr uint8_t MagicWord = 0xDC;
  union {
    uint64_t word0 = 0; ///< bits 0 - 63
    struct {
      uint8_t version : 8;          ///< version
      uint8_t packetID : 8;         ///< packet id
      uint8_t errorCode : 8;        ///< errors
      uint8_t magicWord : 8;        ///< magic word
      uint32_t heartbeatOrbit : 32; ///< first heart beat timing of the package
    };
  };
  union {
    uint64_t word1 = 0; ///< bits 64 - 127
    struct {
      uint16_t heartbeatBC : 16; ///< first BC id of the package
      uint16_t unused1 : 16;     ///< reserved
      uint32_t unused2 : 32;     ///< reserved
    };
  };
  union {
    uint64_t word3 = 0; ///< bits 128 - 191
    struct {
      uint64_t unused3 : 64; ///< reserved
    };
  };
  union {
    uint64_t word4 = 0; ///< bits 192 - 255
    struct {
      uint64_t unused4 : 64; ///< reserved
    };
  };
};

/// CMV single data container
struct Data {
  uint16_t cmv{0}; ///< 16-bit signed fixed point value: bit 15 = sign (1=positive, 0=negative), bits 14-0 = I8F7 magnitude

  uint16_t getCMV() const { return cmv; }      ///< raw 16-bit integer representation
  void setCMV(uint16_t value) { cmv = value; } ///< set raw 16-bit integer representation

  // Decode to float: sign-magnitude with 7 fractional bits, range ±255.992
  float getCMVFloat() const
  {
    const bool positive = (cmv >> 15) & 1;          // bit 15: sign (1=positive, 0=negative)
    const float magnitude = (cmv & 0x7FFF) / 128.f; // lower 15 bits, shift right by 7 (divide by 2^7)
    return positive ? magnitude : -magnitude;
  }

  // Encode from float: truncates magnitude to 15 bits, range ±255.992
  void setCMVFloat(float value)
  {
    const bool positive = (value >= 0.f);
    const uint16_t magnitude = static_cast<uint16_t>(
                                 std::lround(std::abs(value) * 128.f)) &
                               0x7FFF;
    cmv = (positive ? 0x8000 : 0x0000) | magnitude;
  }
};

/// CMV full data container: one packet carries NTimeBinsPerPacket CMV values followed by padding
/// Layout: Header (32 bytes) + Data[NTimeBinsPerPacket] (7128 bytes) + padding (8 bytes) = 7168 bytes total (224 * 32 = 7168)
/// The padding bytes at the end of the data array are rubbish/unused and must not be interpreted as CMV values
struct Container {
  Header header;                       ///< CMV data header
  Data data[NTimeBinsPerPacket];       ///< data values
  uint8_t padding[DataPaddingBytes]{}; ///< trailing padding to align data to 32-byte boundary

  // Header and data accessors
  const Header& getHeader() const { return header; }
  Header& getHeader() { return header; }

  const Data* getData() const { return data; }
  Data* getData() { return data; }

  // Per timebin CMV accessors
  uint16_t getCMV(uint32_t timeBin) const { return data[timeBin].getCMV(); }
  void setCMV(uint32_t timeBin, uint16_t value) { data[timeBin].setCMV(value); }

  float getCMVFloat(uint32_t timeBin) const { return data[timeBin].getCMVFloat(); }
  void setCMVFloat(uint32_t timeBin, float value) { data[timeBin].setCMVFloat(value); }
};

} // namespace o2::tpc::cmv

#endif
