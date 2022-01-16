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

/// @file   SIMDDecoder.h
/// @author Michael Lettrich
/// @since  2020-04-06
/// @brief  Decoder - decode a rANS encoded state back into source symbols

#ifndef RANS_SIMDDECODER_H
#define RANS_SIMDDECODER_H

#include <cstddef>
#include <type_traits>
#include <iostream>
#include <iomanip>
#include <memory>

#include <fairlogger/Logger.h>

#include "rANS/FrequencyTable.h"
#include "rANS/internal/backend/cpp/DecoderSymbol.h"
#include "rANS/internal/ReverseSymbolLookupTable.h"
#include "rANS/internal/SymbolTable.h"
#include "rANS/internal/backend/simd/Decoder.h"
#include "rANS/internal/DecoderBase.h"
#include "rANS/internal/helper.h"

namespace o2
{
namespace rans
{

template <typename coder_T,
          typename stream_T,
          typename source_T,
          uint8_t nStreams_V = 4,
          uint8_t nHardwareStreams_V = 2>
class SIMDDecoder : public internal::DecoderBase<coder_T, stream_T, source_T>
{
 public:
  using internal::DecoderBase<coder_T, stream_T, source_T>::DecoderBase;

  template <typename stream_IT,
            typename source_IT,
            std::enable_if_t<internal::isCompatibleIter_v<stream_T, stream_IT>, bool> = true>
  void process(stream_IT inputEnd, source_IT outputBegin, size_t messageLength) const;

 private:
  using ransDecoder_t = internal::simd::Decoder<coder_T, stream_T>;
};

template <typename coder_T, typename stream_T, typename source_T, uint8_t nStreams_V, uint8_t nHardwareStreams_V>
template <typename stream_IT,
          typename source_IT,
          std::enable_if_t<internal::isCompatibleIter_v<stream_T, stream_IT>, bool>>
void SIMDDecoder<coder_T, stream_T, source_T, nStreams_V, nHardwareStreams_V>::process(stream_IT inputEnd, source_IT outputBegin, size_t messageLength) const
{
  using namespace internal;
  LOG(trace) << "start decoding";
  RANSTimer t;
  t.start();

  if (messageLength == 0) {
    LOG(warning) << "Empty message passed to decoder, skipping decode process";
    return;
  }

  stream_IT inputIter = inputEnd;
  source_IT it = outputBegin;

  // make Iter point to the last last element
  --inputIter;

  std::vector<ransDecoder_t> decoders{nStreams_V, ransDecoder_t{this->getSymbolTablePrecision()}};
#pragma GCC unroll 4
  for (size_t i = 0; i < nStreams_V; ++i) {
    inputIter = decoders[i].init(inputIter);
  }

  const size_t nLoops = messageLength / nStreams_V;
  const size_t nLoopRemainder = messageLength % nStreams_V;

  for (size_t i = 0; i < nLoops; ++i) {
#pragma GCC unroll 4
    for (size_t i = 0; i < nStreams_V; ++i) {
      const source_T symbol = this->mReverseLUT[decoders[i].get()];
      *it++ = symbol;
      inputIter = decoders[i].advanceSymbol(inputIter, this->mSymbolTable[symbol]);
    }
  }

  for (size_t i = 0; i < nLoopRemainder; ++i) {
    const source_T symbol = this->mReverseLUT[decoders[i].get()];
    *it++ = symbol;
    inputIter = decoders[i].advanceSymbol(inputIter, this->mSymbolTable[symbol]);
  }

  t.stop();
  LOG(debug1) << "Decoder::" << __func__ << " { DecodedSymbols: " << messageLength << ","
              << "processedBytes: " << messageLength * sizeof(source_T) << ","
              << " inclusiveTimeMS: " << t.getDurationMS() << ","
              << " BandwidthMiBPS: " << std::fixed << std::setprecision(2) << (messageLength * sizeof(source_T) * 1.0) / (t.getDurationS() * 1.0 * (1 << 20)) << "}";

  LOG(trace) << "done decoding";
}
} // namespace rans
} // namespace o2

#endif /* RANS_SIMDDECODER_H */
