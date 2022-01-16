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

/// @file   Decoder.h
/// @author Michael Lettrich
/// @since  2020-04-06
/// @brief  Decoder - decode a rANS encoded state back into source symbols

#ifndef RANS_LITERALSIMDDECODER_H
#define RANS_LITERALSIMDDECODER_H

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
class LiteralSIMDDecoder : public internal::DecoderBase<coder_T, stream_T, source_T>
{
 public:
  using internal::DecoderBase<coder_T, stream_T, source_T>::DecoderBase;

  template <typename stream_IT,
            typename source_IT,
            std::enable_if_t<internal::isCompatibleIter_v<stream_T, stream_IT>, bool> = true>
  void process(stream_IT inputEnd, source_IT outputBegin, size_t messageLength, std::vector<source_T>& literals) const;

 private:
  using ransDecoder_t = internal::simd::Decoder<coder_T, stream_T>;
};

template <typename coder_T, typename stream_T, typename source_T, uint8_t nStreams_V, uint8_t nHardwareStreams_V>
template <typename stream_IT,
          typename source_IT,
          std::enable_if_t<internal::isCompatibleIter_v<stream_T, stream_IT>, bool>>
void LiteralSIMDDecoder<coder_T, stream_T, source_T, nStreams_V, nHardwareStreams_V>::process(stream_IT inputEnd, source_IT outputBegin, size_t messageLength, std::vector<source_T>& literals) const
{
  using namespace internal;
  LOG(trace) << "start decoding";
  RANSTimer t;
  t.start();

#ifdef O2_RANS_PRINT_PROCESSED_DATA
  JSONArrayLogger<source_T> arrayLogger{};
#endif

  if (messageLength == 0) {
    LOG(warning) << "Empty message passed to decoder, skipping decode process";
    return;
  }

  stream_IT inputIter = inputEnd;
  source_IT it = outputBegin;

  auto decode = [&, this](ransDecoder_t& decoder) {
    const auto cumul = decoder.get();
    const auto streamSymbol = (this->mReverseLUT)[cumul];
    source_T symbol = streamSymbol;
    if (this->mSymbolTable.isEscapeSymbol(streamSymbol)) {
      symbol = literals.back();
      literals.pop_back();
    }

#ifdef O2_RANS_PRINT_PROCESSED_DATA
    arrayLogger << symbol;
#endif

    return std::make_tuple(symbol, decoder.advanceSymbol(inputIter, (this->mSymbolTable)[streamSymbol]));
  };

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
      std::tie(*it++, inputIter) = decode(decoders[i]);
    }
  }

  for (size_t i = 0; i < nLoopRemainder; ++i) {
    std::tie(*it++, inputIter) = decode(decoders[i]);
  }
  t.stop();

#ifdef O2_RANS_PRINT_PROCESSED_DATA
  LOG(info) << "decoderOutput:" << arrayLogger;
#endif

  LOG(debug1) << "Decoder::" << __func__ << " { DecodedSymbols: " << messageLength << ","
              << "processedBytes: " << messageLength * sizeof(source_T) << ","
              << " inclusiveTimeMS: " << t.getDurationMS() << ","
              << " BandwidthMiBPS: " << std::fixed << std::setprecision(2) << (messageLength * sizeof(source_T) * 1.0) / (t.getDurationS() * 1.0 * (1 << 20)) << "}";

  LOG(trace) << "done decoding";
}
} // namespace rans
} // namespace o2

#endif /* RANS_LITERALSIMDDECODER_H */
