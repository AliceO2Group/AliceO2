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

/// \file Metadata.h
/// \brief Metadata required to decode a Block

///  The Metadata required to decoded the CTF of particular detector.

#ifndef ALICEO2_METADATA_H
#define ALICEO2_METADATA_H

#include <cstddef>
#include <Rtypes.h>

namespace o2::ctf
{

struct Metadata {
  enum class OptStore : uint8_t { // describe how the store the data described by this metadata
    EENCODE,                      // entropy encoding applied
    ROOTCompression,              // original data repacked to array with slot-size = streamSize and saved with root compression
    NONE,                         // original data repacked to array with slot-size = streamSize and saved w/o compression
    NODATA,                       // no data was provided
    PACK,                         // use Bitpacking
    EENCODE_OR_PACK               // decide at runtime if to encode or pack
  };
  uint8_t nStreams = 0;              // Amount of concurrent Streams used by the encoder. only used by rANS version >=1.
  size_t messageLength = 0;          // Message length (multiply with messageWordSize to get size in Bytes).
  size_t nLiterals = 0;              // Number of samples that were stored as literals.
  uint8_t messageWordSize = 0;       // size in Bytes of a symbol in the encoded message.
  uint8_t coderType = 0;             // what type of CTF Coder is used? (32 vs 64 bit coders).
  uint8_t streamSize = 0;            // how many Bytes is the rANS encoder emmiting during a stream-out step.
  uint8_t probabilityBits = 0;       // The encoder renormed the distribution of source symbols to sum up to 2^probabilityBits.
  OptStore opt = OptStore::EENCODE;  // The type of storage operation that was conducted.
  int32_t min = 0;                   // min symbol of the source dataset.
  int32_t max = 0;                   // max symbol of the source dataset.
  int32_t literalsPackingOffset = 0; // Offset from 0 used for bit packing of literal (incompressible) symbols. only used by rANS version >=1.
  uint8_t literalsPackingWidth = 0;  // Amount of bits used to pack literal (incompressible) symbols. only used by rANS version >=1.
  int nDictWords = 0;                // Amount of words used to store the encoding dictionary.
  int nDataWords = 0;                // Amount of words used to store the actual data.
  int nLiteralWords = 0;             // Amount of words used to store literal (incompressible) samples.

  size_t getUncompressedSize() const { return messageLength * messageWordSize; }
  size_t getCompressedSize() const { return (nDictWords + nDataWords + nLiteralWords) * streamSize; }
  void clear()
  {
    nStreams = 0;
    messageLength = 0;
    nLiterals = 0;
    messageWordSize = 0;
    coderType = 0;
    streamSize = 0;
    probabilityBits = 0;
    min = 0;
    max = 0;
    literalsPackingOffset = 0;
    literalsPackingWidth = 0;
    nDictWords = 0;
    nDataWords = 0;
    nLiteralWords = 0;
  }
  ClassDefNV(Metadata, 3);
};

namespace detail
{

template <typename source_T, typename state_T, typename stream_T>
[[nodiscard]] inline constexpr Metadata makeMetadataRansCompat(size_t nStreams, size_t messageLength,
                                                               size_t nLiterals, size_t symbolTablePrecision,
                                                               source_T min, source_T max, size_t dictWords,
                                                               size_t dataWords, size_t literalWords) noexcept
{
  return Metadata{
    static_cast<uint8_t>(nStreams),
    messageLength,
    nLiterals,
    static_cast<uint8_t>(sizeof(source_T)),
    static_cast<uint8_t>(sizeof(state_T)),
    static_cast<uint8_t>(sizeof(stream_T)),
    static_cast<uint8_t>(symbolTablePrecision),
    Metadata::OptStore::EENCODE,
    static_cast<int32_t>(min),
    static_cast<int32_t>(max),
    static_cast<int32_t>(0),
    static_cast<uint8_t>(sizeof(source_T)),
    static_cast<int32_t>(dictWords),
    static_cast<int32_t>(dataWords),
    static_cast<int32_t>(literalWords)};
};

template <typename source_T>
[[nodiscard]] inline constexpr Metadata makeMetadataRansDict(size_t symbolTablePrecision, source_T min,
                                                             source_T max, size_t dictWords, ctf::Metadata::OptStore optStore) noexcept
{
  return Metadata{
    static_cast<uint8_t>(0),
    static_cast<uint8_t>(0),
    static_cast<uint8_t>(0),
    static_cast<uint8_t>(sizeof(source_T)),
    static_cast<uint8_t>(0),
    static_cast<uint8_t>(0),
    static_cast<uint8_t>(symbolTablePrecision),
    optStore,
    static_cast<int32_t>(min),
    static_cast<int32_t>(max),
    static_cast<int32_t>(0),
    static_cast<uint8_t>(0),
    static_cast<int32_t>(dictWords),
    static_cast<int32_t>(0),
    static_cast<int32_t>(0)};
};

template <typename source_T, typename state_T, typename stream_T>
[[nodiscard]] inline constexpr Metadata makeMetadataRansV1(size_t nStreams, size_t streamingLowerBound, size_t messageLength,
                                                           size_t nLiterals, size_t symbolTablePrecision,
                                                           source_T dictMin, source_T dictMax,
                                                           source_T literalsOffset, size_t literalsPackingWidth, size_t dictWords,
                                                           size_t dataWords, size_t literalWords) noexcept
{
  return Metadata{
    static_cast<uint8_t>(nStreams),
    messageLength,
    nLiterals,
    static_cast<uint8_t>(sizeof(source_T)),
    static_cast<uint8_t>(sizeof(state_T)),
    static_cast<uint8_t>(streamingLowerBound),
    static_cast<uint8_t>(symbolTablePrecision),
    Metadata::OptStore::EENCODE,
    static_cast<int32_t>(dictMin),
    static_cast<int32_t>(dictMax),
    static_cast<int32_t>(literalsOffset),
    static_cast<uint8_t>(literalsPackingWidth),
    static_cast<int32_t>(dictWords),
    static_cast<int32_t>(dataWords),
    static_cast<int32_t>(literalWords)};
};

template <typename source_T>
[[nodiscard]] inline constexpr Metadata makeMetadataPack(size_t messageLength, size_t packingWidth,
                                                         source_T packingOffset, size_t dataWords) noexcept
{
  return Metadata{
    static_cast<uint8_t>(1),
    messageLength,
    static_cast<size_t>(0),
    static_cast<uint8_t>(sizeof(source_T)),
    static_cast<uint8_t>(0),
    static_cast<uint8_t>(0),
    static_cast<uint8_t>(packingWidth),
    Metadata::OptStore::PACK,
    static_cast<int32_t>(packingOffset),
    static_cast<int32_t>(0),
    static_cast<int32_t>(0),
    static_cast<uint8_t>(0),
    static_cast<int32_t>(0),
    static_cast<int32_t>(dataWords),
    static_cast<int32_t>(0)};
};

template <typename source_T, typename buffer_T>
[[nodiscard]] inline constexpr Metadata makeMetadataStore(size_t messageLength, Metadata::OptStore opStore, size_t dataWords) noexcept
{
  return Metadata{
    static_cast<uint8_t>(0),
    messageLength,
    static_cast<size_t>(0),
    static_cast<uint8_t>(sizeof(source_T)),
    static_cast<uint8_t>(0),
    static_cast<uint8_t>(sizeof(buffer_T)),
    static_cast<uint8_t>(0),
    opStore,
    static_cast<int32_t>(0),
    static_cast<int32_t>(0),
    static_cast<int32_t>(0),
    static_cast<int32_t>(0),
    static_cast<int32_t>(0),
    static_cast<int32_t>(dataWords),
    static_cast<int32_t>(0)};
};

} // namespace detail
} // namespace o2::ctf

#endif