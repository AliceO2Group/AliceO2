// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   rans.h
/// @author Michael Lettrich
/// @since  2019-05-21
/// @brief  header for public api

#ifndef RANS_RANS_H
#define RANS_RANS_H

#include "rANS/FrequencyTable.h"
#include "rANS/Encoder.h"
#include "rANS/Decoder.h"
#include "rANS/DedupEncoder.h"
#include "rANS/DedupDecoder.h"
#include "rANS/LiteralEncoder.h"
#include "rANS/LiteralDecoder.h"
#include "rANS/internal/helper.h"

namespace o2
{
namespace rans
{
template <typename source_T>
using Encoder32 = Encoder<uint32_t, uint8_t, source_T>;
template <typename source_T>
using Encoder64 = Encoder<uint64_t, uint32_t, source_T>;

template <typename source_T>
using Decoder32 = Decoder<uint32_t, uint8_t, source_T>;
template <typename source_T>
using Decoder64 = Decoder<uint64_t, uint32_t, source_T>;

template <typename source_T>
using LiteralEncoder32 = LiteralEncoder<uint32_t, uint8_t, source_T>;
template <typename source_T>
using LiteralEncoder64 = LiteralEncoder<uint64_t, uint32_t, source_T>;

template <typename source_T>
using LiteralDecoder32 = LiteralDecoder<uint32_t, uint8_t, source_T>;
template <typename source_T>
using LiteralDecoder64 = LiteralDecoder<uint64_t, uint32_t, source_T>;

template <typename source_T>
using DedupEncoder32 = DedupEncoder<uint32_t, uint8_t, source_T>;
template <typename source_T>
using DedupEncoder64 = DedupEncoder<uint64_t, uint32_t, source_T>;

template <typename source_T>
using DedupDecoder32 = DedupDecoder<uint32_t, uint8_t, source_T>;
template <typename source_T>
using DedupDecoder64 = DedupDecoder<uint64_t, uint32_t, source_T>;

inline size_t calculateMaxBufferSize(size_t num, size_t rangeBits, size_t sizeofStreamT)
{
  //  // RS: w/o safety margin the o2-test-ctf-io produces an overflow in the Encoder::process
  //  constexpr size_t SaferyMargin = 16;
  //  return std::ceil(1.20 * (num * rangeBits * 1.0) / (sizeofStreamT * 8.0)) + SaferyMargin;
  return num * sizeofStreamT;
}

} // namespace rans
} // namespace o2

#endif /* RANS_RANS_H */
