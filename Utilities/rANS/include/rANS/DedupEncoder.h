// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Encoder.h
/// @author Michael Lettrich
/// @since  2020-04-06
/// @brief  Encoder - code symbol into a rANS encoded state

#ifndef INCLUDE_RANS_DEDUPENCODER_H_
#define INCLUDE_RANS_DEDUPENCODER_H_

#include <memory>
#include <algorithm>
#include <iomanip>
#include <map>
#include <cstdint>
#include <string>

#include <fairlogger/Logger.h>
#include <stdexcept>

#include "rANS/internal/EncoderBase.h"
#include "rANS/internal/EncoderSymbol.h"
#include "rANS/internal/helper.h"
#include "rANS/internal/SymbolTable.h"

namespace o2
{
namespace rans
{

template <typename coder_T, typename stream_T, typename source_T>
class DedupEncoder : public internal::EncoderBase<coder_T, stream_T, source_T>
{
 public:
  //inherit constructors;
  using internal::EncoderBase<coder_T, stream_T, source_T>::EncoderBase;

  using duplicatesMap_t = std::map<uint32_t, uint32_t>;

  template <typename stream_IT, typename source_IT, std::enable_if_t<internal::isCompatibleIter_v<source_T, source_IT>, bool> = true>
  stream_IT process(source_IT inputBegin, source_IT inputEnd, stream_IT outputBegin, duplicatesMap_t& duplicates) const;

 private:
  using ransCoder_t = typename internal::EncoderBase<coder_T, stream_T, source_T>::ransCoder_t;
};

template <typename coder_T, typename stream_T, typename source_T>
template <typename stream_IT, typename source_IT, std::enable_if_t<internal::isCompatibleIter_v<source_T, source_IT>, bool>>
stream_IT DedupEncoder<coder_T, stream_T, source_T>::process(source_IT inputBegin, source_IT inputEnd, stream_IT outputBegin, duplicatesMap_t& duplicates) const
{
  using namespace internal;
  LOG(trace) << "start encoding";
  RANSTimer t;
  t.start();

  if (inputBegin == inputEnd) {
    LOG(warning) << "passed empty message to encoder, skip encoding";
    return outputBegin;
  }
  ransCoder_t rans{this->mSymbolTablePrecission};

  stream_IT outputIter = outputBegin;
  source_IT inputIT = inputEnd;

  const auto inputBufferSize = std::distance(inputBegin, inputEnd);

  auto encode = [&inputBegin, &duplicates, this](source_IT symbolIter, stream_IT outputIter, ransCoder_t& coder) {
    const source_T symbol = *symbolIter;
    const auto& encoderSymbol = (this->mSymbolTable)[symbol];

    // dedup step:
    auto dedupIT = symbolIter;
    //advance on source by one.
    --dedupIT;
    size_t numDuplicates = 0;

    // find out how many duplicates we have
    while (*dedupIT == symbol) {
      --dedupIT;
      ++numDuplicates;
    }

    // if we have a duplicate treat it.
    if (numDuplicates > 0) {
      const auto pos = std::distance(inputBegin, symbolIter) - 1;
      LOG(trace) << "pos[" << pos << "]: found " << numDuplicates << " duplicates of symbol " << (char)symbol;
      duplicates.emplace(pos, numDuplicates);
    }

    return std::pair(++dedupIT, coder.putSymbol(outputIter, encoderSymbol));
  };

  while (inputIT != inputBegin) { // NB: working in reverse!
    std::tie(inputIT, outputIter) = encode(--inputIT, outputIter, rans);
  }
  outputIter = rans.flush(outputIter);
  // first iterator past the range so that sizes, distances and iterators work correctly.
  ++outputIter;

  t.stop();
  LOG(debug1) << "Encoder::" << __func__ << " {ProcessedBytes: " << inputBufferSize * sizeof(source_T) << ","
              << " inclusiveTimeMS: " << t.getDurationMS() << ","
              << " BandwidthMiBPS: " << std::fixed << std::setprecision(2) << (inputBufferSize * sizeof(source_T) * 1.0) / (t.getDurationS() * 1.0 * (1 << 20)) << "}";

// advanced diagnostics for debug builds
#if !defined(NDEBUG)

  const auto inputBufferSizeB = inputBufferSize * sizeof(source_T);

  LOG(debug2)
    << "EncoderProperties: {"
    << "sourceTypeB: " << sizeof(source_T) << ", "
    << "streamTypeB: " << sizeof(stream_T) << ", "
    << "coderTypeB: " << sizeof(coder_T) << ", "
    << "probabilityBits: " << this->mSymbolTablePrecission << ", "
    << "inputBufferSizeB: " << inputBufferSizeB << "}";
#endif

  LOG(trace) << "done encoding";

  return outputIter;
};

} // namespace rans
} // namespace o2

#endif /* INCLUDE_RANS_DEDUPENCODER_H_ */
