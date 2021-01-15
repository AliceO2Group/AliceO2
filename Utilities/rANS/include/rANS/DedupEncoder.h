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

#include "Encoder.h"

#include <memory>
#include <algorithm>
#include <iomanip>
#include <map>
#include <cstdint>
#include <string>

#include <fairlogger/Logger.h>
#include <stdexcept>

#include "internal/EncoderSymbol.h"
#include "internal/helper.h"
#include "internal/SymbolTable.h"

namespace o2
{
namespace rans
{

template <typename coder_T, typename stream_T, typename source_T>
class DedupEncoder : public Encoder<coder_T, stream_T, source_T>
{
  //inherit constructors;
  using Encoder<coder_T, stream_T, source_T>::Encoder;

 public:
  using duplicatesMap_t = std::map<uint32_t, uint32_t>;

  template <typename stream_IT, typename source_IT, std::enable_if_t<internal::isCompatibleIter_v<stream_T, stream_IT> && internal::isCompatibleIter_v<source_T, source_IT>, bool> = true>
  const stream_IT process(const stream_IT outputBegin, const stream_IT outputEnd,
                          const source_IT inputBegin, source_IT inputEnd, duplicatesMap_t& duplicates) const;
};

template <typename coder_T, typename stream_T, typename source_T>
template <typename stream_IT, typename source_IT, std::enable_if_t<internal::isCompatibleIter_v<stream_T, stream_IT> && internal::isCompatibleIter_v<source_T, source_IT>, bool>>
const stream_IT DedupEncoder<coder_T, stream_T, source_T>::process(const stream_IT outputBegin, const stream_IT outputEnd, const source_IT inputBegin, const source_IT inputEnd, duplicatesMap_t& duplicates) const
{
  using namespace internal;
  using ransCoder = internal::Encoder<coder_T, stream_T>;
  LOG(trace) << "start encoding";
  RANSTimer t;
  t.start();

  static_assert(std::is_same<typename std::iterator_traits<source_IT>::value_type, source_T>::value);
  static_assert(std::is_same<typename std::iterator_traits<stream_IT>::value_type, stream_T>::value);

  if (inputBegin == inputEnd) {
    LOG(warning) << "passed empty message to encoder, skip encoding";
    return outputEnd;
  }

  if (outputBegin == outputEnd) {
    const std::string errorMessage("Unallocated encode buffer passed to encoder. Aborting");
    LOG(error) << errorMessage;
    throw std::runtime_error(errorMessage);
  }

  ransCoder rans;

  stream_IT outputIter = outputBegin;
  source_IT inputIT = inputEnd;

  const auto inputBufferSize = std::distance(inputBegin, inputEnd);

  auto encode = [&inputBegin, &duplicates, this](source_IT symbolIter, stream_IT outputIter, ransCoder& coder) {
    const source_T symbol = *symbolIter;
    const auto& encoderSymbol = (*this->mSymbolTable)[symbol];

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
      LOG(trace) << "pos[" << std::distance(inputBegin, symbolIter) - 1 << "]: found " << numDuplicates << " duplicates of symbol " << (char)symbol;
      duplicates.emplace(std::distance(inputBegin, symbolIter) - 1, numDuplicates);
    }

    return std::tuple(++dedupIT, coder.putSymbol(outputIter, encoderSymbol, this->mProbabilityBits));
  };

  while (inputIT != inputBegin) { // NB: working in reverse!
    std::tie(inputIT, outputIter) = encode(--inputIT, outputIter, rans);
    assert(outputIter < outputEnd);
  }
  outputIter = rans.flush(outputIter);
  // first iterator past the range so that sizes, distances and iterators work correctly.
  ++outputIter;

  assert(!(outputIter > outputEnd));

  // deal with overflow
  if (outputIter > outputEnd) {
    const std::string exceptionText = [&]() {
      std::stringstream ss;
      ss << __func__ << " detected overflow in encode buffer: allocated:" << std::distance(outputBegin, outputEnd) << ", used:" << std::distance(outputBegin, outputIter);
      return ss.str();
    }();

    LOG(error) << exceptionText;
    throw std::runtime_error(exceptionText);
  }

  t.stop();
  LOG(debug1) << "Encoder::" << __func__ << " {ProcessedBytes: " << inputBufferSize * sizeof(source_T) << ","
              << " inclusiveTimeMS: " << t.getDurationMS() << ","
              << " BandwidthMiBPS: " << std::fixed << std::setprecision(2) << (inputBufferSize * sizeof(source_T) * 1.0) / (t.getDurationS() * 1.0 * (1 << 20)) << "}";

// advanced diagnostics for debug builds
#if !defined(NDEBUG)

  const auto inputBufferSizeB = inputBufferSize * sizeof(source_T);
  const auto outputBufferSizeB = std::distance(outputBegin, outputIter) * sizeof(stream_T);

  LOG(debug2) << "EncoderProperties: {"
              << "sourceTypeB: " << sizeof(source_T) << ", "
              << "streamTypeB: " << sizeof(stream_T) << ", "
              << "coderTypeB: " << sizeof(coder_T) << ", "
              << "probabilityBits: " << this->mProbabilityBits << ", "
              << "inputBufferSizeB: " << inputBufferSizeB << ", "
              << "outputBufferSizeB: " << outputBufferSizeB << ", "
              << "compressionFactor: " << std::fixed << std::setprecision(2) << static_cast<double>(inputBufferSizeB) / static_cast<double>(outputBufferSizeB) << "}";
#endif

  LOG(trace) << "done encoding";

  return outputIter;
};

} // namespace rans
} // namespace o2

#endif /* INCLUDE_RANS_DEDUPENCODER_H_ */
