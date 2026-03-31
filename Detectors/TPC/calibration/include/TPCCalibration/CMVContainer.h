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

/// @file   CMVContainer.h
/// @author Tuba Gündem, tuba.gundem@cern.ch
/// @brief  Structs for storing CMVs to the CCDB

#ifndef ALICEO2_TPC_CMVCONTAINER_H_
#define ALICEO2_TPC_CMVCONTAINER_H_

#include <string>
#include <memory>
#include <vector>
#include <cstdint>

#include "TTree.h"
#include "TPCBase/CRU.h"
#include "DataFormatsTPC/CMV.h"

namespace o2::tpc
{

struct CMVPerTF; // forward declaration

/// Delta+zigzag+varint compressed CMV data for one TF across all CRUs
/// Produced by CMVPerTF::compress(), restored with decompress()
/// Each TTree entry corresponds to one CMVPerTFCompressed object (one TF)
struct CMVPerTFCompressed {
  uint32_t firstOrbit{0}; ///< First orbit of this TF (copied from CMVPerTF)
  uint16_t firstBC{0};    ///< First bunch crossing of this TF (copied from CMVPerTF)

  /// Delta+zigzag+varint encoded CMV values
  /// Layout: CRU-major, time-minor; delta is reset to zero at each CRU boundary
  std::vector<uint8_t> mCompressedData;

  /// Restore a CMVPerTF from this compressed object into *cmv (must not be null)
  void decompress(CMVPerTF* cmv) const;

 private:
  static uint16_t signedToCmv(int32_t val);                               ///< Signed integer -> sign-magnitude uint16_t
  static int32_t zigzagDecode(uint32_t value);                            ///< Zigzag decode
  static uint32_t decodeVarint(const uint8_t*& data, const uint8_t* end); ///< Varint decode

 public:
  ClassDefNV(CMVPerTFCompressed, 1)
};

/// CMV data for one TF across all CRUs
/// Raw 16-bit CMV values are stored in a flat C array indexed as [cru * NTimeBinsPerTF + timeBin]
/// CRU::MaxCRU and cmv::NTimeBinsPerTF are compile-time constants, so no dynamic allocation is needed
struct CMVPerTF {
  uint32_t firstOrbit{0}; ///< First orbit of this TF, from heartbeatOrbit of the first CMV packet
  uint16_t firstBC{0};    ///< First bunch crossing of this TF, from heartbeatBC of the first CMV packet

  // Raw 16-bit CMV values, flat array indexed as [cru * NTimeBinsPerTF + timeBin]
  uint16_t mDataPerTF[CRU::MaxCRU * cmv::NTimeBinsPerTF]{};

  /// Return the raw 16-bit CMV value for a given CRU and timebin within this TF
  uint16_t getCMV(const int cru, const int timeBin) const;

  /// Return the float CMV value for a given CRU and timebin within this TF
  float getCMVFloat(const int cru, const int timeBin) const;

  /// Zero out raw CMV values whose float magnitude is below threshold (default 1.0 ADC)
  /// This converts the sign-magnitude raw value to 0x0000 for all entries with |float value| < threshold
  void zeroSmallValues(float threshold = 1.0f);

  /// Compress this object into a CMVPerTFCompressed using delta+zigzag+varint encoding
  CMVPerTFCompressed compress() const;

  /// Serialise into a TTree; each Fill() call appends one entry (one TF)
  std::unique_ptr<TTree> toTTree() const;

  /// Write the TTree to a ROOT file
  static void writeToFile(const std::string& filename, const std::unique_ptr<TTree>& tree);

 private:
  static int32_t cmvToSigned(uint16_t raw);                                ///< Sign-magnitude uint16_t → signed integer
  static uint32_t zigzagEncode(int32_t value);                             ///< Zigzag encode
  static void encodeVarintInto(uint32_t value, std::vector<uint8_t>& out); ///< Varint encode

 public:
  ClassDefNV(CMVPerTF, 1)
};

} // namespace o2::tpc

#endif // ALICEO2_TPC_CMVCONTAINER_H_