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
/// - 16-bit CMV value: [CMV:16]

#ifndef ALICEO2_DATAFORMATSTPC_CMV_H
#define ALICEO2_DATAFORMATSTPC_CMV_H

#include <cstdint>

namespace o2::tpc::cmv
{

static constexpr uint32_t NTimeBinsPerPacket = 3564;                                 ///< number of time bins (covering 8 heartbeats)
static constexpr uint32_t NPacketsPerTFPerCRU = 4;                                   ///< 4 packets per timeframe
static constexpr uint32_t NTimeBinsPerTF = NTimeBinsPerPacket * NPacketsPerTFPerCRU; ///< maximum number of timebins per timeframe (14256)
static constexpr uint32_t SignificantBits = 2;                                       ///< number of bits used for floating point precision
static constexpr float FloatConversion = 1.f / float(1 << SignificantBits);          ///< conversion factor from integer representation to float

/// Header definition of the CMVs
struct Header {
  static constexpr uint8_t MagicWord = 0xDC;
  union {
    uint32_t word0 = 0; ///< bits 0 - 31
    struct {
      uint8_t version : 8;   ///< version
      uint8_t packetID : 8;  ///< packet id
      uint8_t errorCode : 8; ///< errors
      uint8_t magicWord : 8; ///< magic word
    };
  };
  union {
    uint32_t word1 = 0; ///< bits 32 - 63
    struct {
      uint32_t heartbeatOrbit : 32; ///< first heart beat timing of the package
    };
  };
  union {
    uint16_t word2 = 0; ///< bits 64 - 79
    struct {
      uint16_t heartbeatBC : 16; ///< first BC id of the package
    };
  };
  union {
    uint16_t word3 = 0; ///< bits 80 - 95
    struct {
      uint16_t unused1 : 16; ///< reserved
    };
  };
  union {
    uint32_t word4 = 0; ///< bits 96 - 127
    struct {
      uint32_t unused2 : 32; ///< reserved
    };
  };
  union {
    uint64_t word5 = 0; ///< bits 128 - 191
    struct {
      uint64_t unused3 : 64; ///< reserved
    };
  };
  union {
    uint64_t word6 = 0; ///< bits 192 - 255
    struct {
      uint64_t unused4 : 64; ///< reserved
    };
  };
};

/// CMV single data container
struct Data {
  uint16_t CMV{0}; ///< 16bit ADC value

  // Raw integer accessors
  uint16_t getCMV() const { return CMV; }
  void setCMV(uint16_t value) { CMV = value; }

  // Float helpers using SignificantBits for fixed-point conversion
  float getCMVFloat() const { return static_cast<float>(CMV) * FloatConversion; }
  void setCMVFloat(float value)
  {
    // round to nearest representable fixed-point value
    setCMV(uint32_t((value + 0.5f * FloatConversion) / FloatConversion));
  }
};

/// CMV full data container: one packet carries NTimeBinsPerPacket
struct Container {
  Header header;                 ///< CMV data header
  Data data[NTimeBinsPerPacket]; ///< data values

  // Header and data accessors
  const Header& getHeader() const { return header; }
  Header& getHeader() { return header; }

  const Data* getData() const { return data; }
  Data* getData() { return data; }

  // Per-time-bin CMV accessors
  uint16_t getCMV(uint32_t timeBin) const { return data[timeBin].getCMV(); }
  void setCMV(uint32_t timeBin, uint16_t value) { data[timeBin].setCMV(value); }

  float getCMVFloat(uint32_t timeBin) const { return data[timeBin].getCMVFloat(); }
  void setCMVFloat(uint32_t timeBin, float value) { data[timeBin].setCMVFloat(value); }
};

} // namespace o2::tpc::cmv

#endif