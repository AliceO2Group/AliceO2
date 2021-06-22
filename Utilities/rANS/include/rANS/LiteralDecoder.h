// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Decoder.h
/// @author Michael Lettrich
/// @since  2020-04-06
/// @brief  Decoder - decode a rANS encoded state back into source symbols

#ifndef RANS_LITERALDECODER_H
#define RANS_LITERALDECODER_H

#include "Decoder.h"

#include <cstddef>
#include <type_traits>
#include <iostream>
#include <iomanip>
#include <string>

#include <fairlogger/Logger.h>

#include "rANS/internal/DecoderSymbol.h"
#include "rANS/internal/ReverseSymbolLookupTable.h"
#include "rANS/internal/SymbolTable.h"
#include "rANS/internal/Decoder.h"
#include "rANS/internal/DecoderBase.h"

namespace o2
{
namespace rans
{

template <typename coder_T, typename stream_T, typename source_T>
class LiteralDecoder : public internal::DecoderBase<coder_T, stream_T, source_T>
{
 public:
  using internal::DecoderBase<coder_T, stream_T, source_T>::DecoderBase;

  template <typename stream_IT, typename source_IT, std::enable_if_t<internal::isCompatibleIter_v<stream_T, stream_IT>, bool> = true>
  void process(stream_IT inputEnd, source_IT outputBegin, size_t messageLength, std::vector<source_T>& literals) const;

 private:
  using ransDecoder_t = typename internal::DecoderBase<coder_T, stream_T, source_T>::ransDecoder_t;
};

template <typename coder_T, typename stream_T, typename source_T>
template <typename stream_IT, typename source_IT, std::enable_if_t<internal::isCompatibleIter_v<stream_T, stream_IT>, bool>>
void LiteralDecoder<coder_T, stream_T, source_T>::process(stream_IT inputEnd, source_IT outputBegin, size_t messageLength, std::vector<source_T>& literals) const
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

  auto decode = [&, this](ransDecoder_t& decoder) {
    const auto cumul = decoder.get();
    const auto streamSymbol = (this->mReverseLUT)[cumul];
    source_T symbol = streamSymbol;
    if (this->mSymbolTable.isEscapeSymbol(streamSymbol)) {
      symbol = literals.back();
      literals.pop_back();
    }

    return std::make_tuple(symbol, decoder.advanceSymbol(inputIter, (this->mSymbolTable)[streamSymbol]));
  };

  // make Iter point to the last last element
  --inputIter;

  ransDecoder_t rans0{this->mSymbolTablePrecission};
  ransDecoder_t rans1{this->mSymbolTablePrecission};
  inputIter = rans0.init(inputIter);
  inputIter = rans1.init(inputIter);

  for (size_t i = 0; i < (messageLength & ~1); i += 2) {
    std::tie(*it++, inputIter) = decode(rans0);
    std::tie(*it++, inputIter) = decode(rans1);
  }

  // last byte, if message length was odd
  if (messageLength & 1) {
    std::tie(*it++, inputIter) = decode(rans0);
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

#endif /* RANS_LITERALDECODER_H */
