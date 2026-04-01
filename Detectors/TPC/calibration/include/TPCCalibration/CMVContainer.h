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

struct CMVPerTF;        // forward declaration
struct CMVPerTFSparse;  // forward declaration
struct CMVPerTFHuffman; // forward declaration

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

/// Sparse-encoded CMV data for one TF across all CRUs
/// Produced by CMVPerTF::compressSparse(), restored with decompress()
/// Each TTree entry corresponds to one CMVPerTFSparse object (one TF)
///
/// Encoding format (stored in mSparseData):
///   For each CRU 0..MaxCRU-1:
///     varint(N)  — number of non-zero timebins in this CRU
///     For each of the N entries (in timebin order):
///       varint(delta)  — absolute timeBin for the first entry; tb − prev_tb for subsequent ones
///       uint16_t(raw)  — raw sign-magnitude CMV value (little-endian, 2 bytes)
struct CMVPerTFSparse {
  uint32_t firstOrbit{0}; ///< First orbit of this TF (copied from CMVPerTF)
  uint16_t firstBC{0};    ///< First bunch crossing of this TF (copied from CMVPerTF)

  /// Sparse-encoded CMV values
  std::vector<uint8_t> mSparseData;

  /// Restore a CMVPerTF from this sparse object into *cmv (must not be null)
  void decompress(CMVPerTF* cmv) const;

 private:
  static uint32_t decodeVarint(const uint8_t*& data, const uint8_t* end); ///< Varint decode

 public:
  ClassDefNV(CMVPerTFSparse, 1)
};

/// Delta+zigzag+canonical Huffman compressed CMV data for one TF across all CRUs
/// Produced by CMVPerTF::compressHuffman(), restored with decompress()
///
/// Serialisation layout (mHuffmanData):
///   4 bytes LE uint32_t : numSymbols — number of distinct zigzag-encoded delta symbols
///   numSymbols * 5 bytes : symbol table (canonical order, sorted by codeLen ASC then symbol ASC):
///       4 bytes LE uint32_t : zigzag-encoded symbol value
///       1 byte              : canonical code length (1..32)
///   8 bytes LE uint64_t : totalBits — number of valid bits in the bitstream
///   ceil(totalBits/8) bytes : bitstream, MSB-first packing within each byte
struct CMVPerTFHuffman {
  uint32_t firstOrbit{0}; ///< First orbit of this TF (copied from CMVPerTF)
  uint16_t firstBC{0};    ///< First bunch crossing of this TF (copied from CMVPerTF)

  /// Huffman-coded payload
  std::vector<uint8_t> mHuffmanData;

  /// Restore a CMVPerTF from this Huffman object into *cmv (must not be null)
  void decompress(CMVPerTF* cmv) const;

  ClassDefNV(CMVPerTFHuffman, 1)
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

  /// Zero out raw CMV values whose float magnitude is below threshold
  /// This converts the sign-magnitude raw value to 0x0000 for all entries with |float value| < threshold
  void zeroSmallValues(float threshold = 1.0f);

  /// Apply dynamic precision reduction: round values to the nearest integer ADC for all values whose rounded magnitude is <= steps
  /// Example: steps=3
  ///   |v| <  0.5 ADC  -> 0
  ///   |v| in [0.5, 1.5) ADC  -> 1.0 ADC
  ///   |v| in [1.5, 2.5) ADC  -> 2.0 ADC
  ///   |v| in [2.5, 3.5) ADC  -> 3.0 ADC
  ///   |v| >= 3.5 ADC         -> unchanged (full precision)
  ///
  /// steps=0 dynamic precision is not applied
  void applyDynamicPrecision(uint16_t steps);

  /// Compress this object into a CMVPerTFCompressed using delta+zigzag+varint encoding
  CMVPerTFCompressed compress() const;

  /// Compress this object into a CMVPerTFSparse storing only non-zero timebins
  CMVPerTFSparse compressSparse() const;

  /// Compress this object using delta+zigzag+canonical-Huffman encoding
  CMVPerTFHuffman compressHuffman() const;

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