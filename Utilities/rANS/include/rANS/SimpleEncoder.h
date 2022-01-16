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

#ifndef RANS_SIMPLEENCODER_H
#define RANS_SIMPLEENCODER_H

#include <memory>
#include <algorithm>
#include <iomanip>

#include <fairlogger/Logger.h>
#include <stdexcept>

#include "rANS/internal/EncoderBase.h"
#include "rANS/internal/backend/cpp/SimpleEncoder.h"
#include "rANS/internal/backend/cpp/Encoder.h"
#include "rANS/internal/backend/cpp/DecoderSymbol.h"
#include "rANS/internal/helper.h"
#include "rANS/internal/SymbolTable.h"
#include "rANS/RenormedFrequencyTable.h"

namespace o2
{
namespace rans
{

template <typename coder_T, typename stream_T, typename source_T>
class SimpleEncoder : public internal::EncoderBase<coder_T, stream_T, source_T>
{
 protected:
  using ransCoder_t = internal::cpp::SimpleEncoder<coder_T, stream_T>;

 public:
  using encoderSymbolTable_t = typename internal::SymbolTable<internal::cpp::DecoderSymbol>;

  using coder_t = coder_T;
  using stream_t = stream_T;
  using source_t = source_T;

  //TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  SimpleEncoder() noexcept {}; //NOLINT
  explicit SimpleEncoder(const RenormedFrequencyTable& frequencies) : mSymbolTable{frequencies} {};

  inline size_t getSymbolTablePrecision() const noexcept { return mSymbolTable.getPrecision(); }
  inline size_t getAlphabetRangeBits() const noexcept { return mSymbolTable.getAlphabetRangeBits(); }
  inline symbol_t getMinSymbol() const noexcept { return mSymbolTable.getMinSymbol(); }
  inline symbol_t getMaxSymbol() const noexcept { return mSymbolTable.getMaxSymbol(); }

  template <typename stream_IT, typename source_IT, std::enable_if_t<internal::isCompatibleIter_v<source_T, source_IT>, bool> = true>
  stream_IT process(source_IT inputBegin, source_IT inputEnd, stream_IT outputBegin);

 private:
  encoderSymbolTable_t mSymbolTable{};
  ransCoder_t mRansCoder{0};
};

template <typename coder_T, typename stream_T, typename source_T>
template <typename stream_IT, typename source_IT, std::enable_if_t<internal::isCompatibleIter_v<source_T, source_IT>, bool>>
stream_IT SimpleEncoder<coder_T, stream_T, source_T>::process(source_IT inputBegin, source_IT inputEnd, stream_IT outputBegin)
{
  using namespace internal;
  LOG(trace) << "start encoding";
  RANSTimer t;
  t.start();

  if (inputBegin == inputEnd) {
    LOG(warning) << "passed empty message to encoder, skip encoding";
    return outputBegin;
  }

  mRansCoder = ransCoder_t{this->getSymbolTablePrecision()};
  auto& rans = mRansCoder;

  stream_IT outputIter = outputBegin;
  source_IT inputIT = inputEnd;

  const auto inputBufferSize = std::distance(inputBegin, inputEnd);

  auto encode = [this](source_IT symbolIter, stream_IT outputIter, ransCoder_t& coder) {
    const source_T symbol = *symbolIter;
    const auto& encoderSymbol = (this->mSymbolTable)[symbol];
    return coder.putSymbol(outputIter, encoderSymbol);
  };

  while (inputIT != inputBegin) { // NB: working in reverse!
    outputIter = encode(--inputIT, outputIter, rans);
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

  LOG(debug2) << "EncoderProperties: {"
              << "sourceTypeB: " << sizeof(source_T) << ", "
              << "streamTypeB: " << sizeof(stream_T) << ", "
              << "coderTypeB: " << sizeof(coder_T) << ", "
              << "probabilityBits: " << this->getSymbolTablePrecision() << ", "
              << "inputBufferSizeB: " << inputBufferSizeB << "}";
#endif

  LOG(trace) << "done encoding";

  return outputIter;
};

} // namespace rans
} // namespace o2

#endif /* RANS_SIMPLEENCODER_H */
