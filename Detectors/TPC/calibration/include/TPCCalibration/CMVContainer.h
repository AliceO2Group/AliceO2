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

struct CMVPerTF;           // forward declaration
struct CMVPerTFCompressed; // forward declaration

/// Bitmask flags describing which encoding stages are applied in CMVPerTFCompressed
struct CMVEncoding {
  static constexpr uint8_t kNone = 0x00;    ///< No compression — raw uint16 values stored flat
  static constexpr uint8_t kSparse = 0x01;  ///< Non-zero positions stored sparsely (varint-encoded deltas)
  static constexpr uint8_t kDelta = 0x02;   ///< Delta coding between consecutive values (dense only)
  static constexpr uint8_t kZigzag = 0x04;  ///< Zigzag encoding of deltas or signed values
  static constexpr uint8_t kVarint = 0x08;  ///< Varint compression of the value stream
  static constexpr uint8_t kHuffman = 0x10; ///< Canonical Huffman compression of the value stream
};

/// Single compressed representation for one TF across all CRUs, stored in a TTree
/// mFlags is a bitmask of CMVEncoding values that fully describes the encoding pipeline
/// mData holds the encoded payload whose binary layout depends on mFlags:
///
///   Dense path (!kSparse):
///     kZigzag absent → N × uint16_t LE  (raw values, CRU-major order)
///     kZigzag + kVarint  → N × varint(zigzag(delta(signed(raw))))
///     kZigzag + kHuffman → [Huffman table] + [bitstream] of zigzag(delta(signed(raw)))
///
///   Sparse path (kSparse):
///     4 bytes LE uint32_t : posStreamSize
///     posStream: for each CRU: varint(N), N × varint(tb_delta)
///     valStream (one entry per non-zero):
///       default          → uint16_t LE raw value
///       kZigzag + kVarint  → varint(zigzag(signed(raw)))
///       kZigzag + kHuffman → [Huffman table] + [bitstream] of zigzag(signed(raw))
struct CMVPerTFCompressed {
  uint32_t firstOrbit{0}; ///< First orbit of this TF
  uint16_t firstBC{0};    ///< First bunch crossing of this TF
  uint8_t mFlags{0};      ///< Bitmask of CMVEncoding values

  std::vector<uint8_t> mData; ///< Encoded payload

  /// Restore a CMVPerTF from this compressed object into *cmv (must not be null)
  void decompress(CMVPerTF* cmv) const;

  /// Serialise into a TTree; each Fill() call appends one entry (one TF)
  std::unique_ptr<TTree> toTTree() const;

 private:
  /// Decode the sparse position stream; advances ptr past the position block
  /// Returns (cru, timeBin) pairs for every non-zero entry, in CRU-major order
  static std::vector<std::pair<int, uint32_t>> decodeSparsePositions(const uint8_t*& ptr, const uint8_t* end);

  /// Decode the value stream into raw uint32_t symbols
  /// Dispatches to Huffman, varint, or raw uint16 based on flags
  static std::vector<uint32_t> decodeValueStream(const uint8_t*& ptr, const uint8_t* end, uint32_t N, uint8_t flags);

  /// Apply inverse zigzag and scatter decoded values into the sparse positions of *cmv
  static void decodeSparseValues(const std::vector<uint32_t>& symbols,
                                 const std::vector<std::pair<int, uint32_t>>& positions,
                                 uint8_t flags, CMVPerTF* cmv);

  /// Apply inverse zigzag and inverse delta, then fill the full dense CMV array in *cmv
  static void decodeDenseValues(const std::vector<uint32_t>& symbols, uint8_t flags, CMVPerTF* cmv);

 public:
  ClassDefNV(CMVPerTFCompressed, 1)
};

/// CMV data for one TF across all CRUs
/// Raw 16-bit CMV values are stored in a flat C array indexed as [cru * NTimeBinsPerTF + timeBin]
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
  void zeroSmallValues(float threshold = 1.0f);

  /// Round values to the nearest integer ADC for all values whose rounded magnitude is <= threshold
  void roundToIntegers(uint16_t threshold);

  /// Quantise |v| with a Gaussian-CDF recovery profile:
  /// Coarse decimal-style precision below and around mean, then a smooth return to the full native I8F7 precision as the magnitude increases with width sigma
  void trimGaussianPrecision(float mean, float sigma);

  /// Compress this object into a CMVPerTFCompressed using the encoding pipeline described by flags
  /// Quantisation (trimGaussianPrecision / roundToIntegers / zeroSmallValues) should be applied to this object before calling compress(); it is not part of the flags pipeline
  CMVPerTFCompressed compress(uint8_t flags) const;

  /// Serialise into a TTree; each Fill() call appends one entry (one TF)
  std::unique_ptr<TTree> toTTree() const;

  /// Write the TTree to a ROOT file
  static void writeToFile(const std::string& filename, const std::unique_ptr<TTree>& tree);

 private:
  static int32_t cmvToSigned(uint16_t raw);                                                              ///< Sign-magnitude uint16_t → signed integer
  static uint16_t quantizeBelowThreshold(uint16_t raw, float quantizationMean, float quantizationSigma); ///< Quantise sub-threshold values with a Gaussian-shaped recovery to full precision
  static uint32_t zigzagEncode(int32_t value);                                                           ///< Zigzag encode
  static void encodeVarintInto(uint32_t value, std::vector<uint8_t>& out);                               ///< Varint encode

 public:
  ClassDefNV(CMVPerTF, 1)
};

} // namespace o2::tpc

#endif // ALICEO2_TPC_CMVCONTAINER_H_
