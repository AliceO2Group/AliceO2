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

#ifndef RANS_ENCODER_H
#define RANS_ENCODER_H

#include <memory>
#include <algorithm>

#include <fairlogger/Logger.h>
#include <stdexcept>

#include "SymbolTable.h"
#include "EncoderSymbol.h"
#include "Coder.h"
#include "helper.h"

namespace o2
{
namespace rans
{
template <typename coder_T, typename stream_T, typename source_T>
class Encoder
{
 private:
  using encoderSymbolTable_t = SymbolTable<EncoderSymbol<coder_T>>;

 public:
  Encoder() = delete;
  ~Encoder() = default;
  Encoder(Encoder&& e) = default;
  Encoder(const Encoder& e);
  Encoder<coder_T, stream_T, source_T>& operator=(const Encoder& e);
  Encoder<coder_T, stream_T, source_T>& operator=(Encoder&& e) = default;

  Encoder(const encoderSymbolTable_t& e, size_t probabilityBits);
  Encoder(encoderSymbolTable_t&& e, size_t probabilityBits);
  Encoder(const SymbolStatistics& stats, size_t probabilityBits);

  template <typename stream_IT, typename source_IT>
  const stream_IT process(const stream_IT outputBegin, const stream_IT outputEnd,
                          const source_IT inputBegin, const source_IT inputEnd) const;

  using coder_t = coder_T;
  using stream_t = stream_T;
  using source_t = source_T;

 private:
  std::unique_ptr<encoderSymbolTable_t> mSymbolTable;
  size_t mProbabilityBits;

  using ransCoder = Coder<coder_T, stream_T>;
};

template <typename coder_T, typename stream_T, typename source_T>
Encoder<coder_T, stream_T, source_T>::Encoder(const Encoder& e) : mSymbolTable(nullptr), mProbabilityBits(e.mProbabilityBits)
{
  mSymbolTable = std::make_unique<encoderSymbolTable_t>(*e.mSymbolTable);
};

template <typename coder_T, typename stream_T, typename source_T>
Encoder<coder_T, stream_T, source_T>& Encoder<coder_T, stream_T, source_T>::operator=(const Encoder& e)
{
  mProbabilityBits = e.mProbabilityBits;
  mSymbolTable = std::make_unique<encoderSymbolTable_t>(*e.mSymbolTable);
  return *this;
};

template <typename coder_T, typename stream_T, typename source_T>
Encoder<coder_T, stream_T, source_T>::Encoder(const encoderSymbolTable_t& e, size_t probabilityBits) : mSymbolTable(nullptr), mProbabilityBits(probabilityBits)
{
  mSymbolTable = std::make_unique<encoderSymbolTable_t>(e);
};

template <typename coder_T, typename stream_T, typename source_T>
Encoder<coder_T, stream_T, source_T>::Encoder(encoderSymbolTable_t&& e, size_t probabilityBits) : mSymbolTable(std::move(e.mSymbolTable)), mProbabilityBits(probabilityBits){};

template <typename coder_T, typename stream_T, typename source_T>
Encoder<coder_T, stream_T, source_T>::Encoder(const SymbolStatistics& stats,
                                              size_t probabilityBits) : mSymbolTable(nullptr), mProbabilityBits(probabilityBits)
{
  RANSTimer t;
  t.start();
  mSymbolTable = std::make_unique<encoderSymbolTable_t>(stats, probabilityBits);
  t.stop();
  LOG(debug1) << "Encoder SymbolTable inclusive time (ms): " << t.getDurationMS();
}

template <typename coder_T, typename stream_T, typename source_T>
template <typename stream_IT, typename source_IT>
const stream_IT Encoder<coder_T, stream_T, source_T>::Encoder::process(
  const stream_IT outputBegin, const stream_IT outputEnd, const source_IT inputBegin, const source_IT inputEnd) const
{
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

  ransCoder rans0, rans1;
  rans0.encInit();
  rans1.encInit();

  stream_IT outputIter = outputBegin;
  source_IT inputIT = inputEnd;

  const auto inputBufferSize = std::distance(inputBegin, inputEnd);

  // odd number of bytes?
  if (inputBufferSize & 1) {
    const coder_T s = *(--inputIT);
    outputIter = rans0.encPutSymbol(outputIter, (*mSymbolTable)[s], mProbabilityBits);
    assert(outputIter < outputEnd);
  }

  while (inputIT > inputBegin) { // NB: working in reverse!
    const coder_T s1 = *(--inputIT);
    const coder_T s0 = *(--inputIT);
    outputIter = rans1.encPutSymbol(outputIter, (*mSymbolTable)[s1], mProbabilityBits);
    outputIter = rans0.encPutSymbol(outputIter, (*mSymbolTable)[s0], mProbabilityBits);
    assert(outputIter < outputEnd);
  }
  outputIter = rans1.encFlush(outputIter);
  outputIter = rans0.encFlush(outputIter);
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
  LOG(debug1) << __func__ << " inclusive time (ms): " << t.getDurationMS();

// advanced diagnostics for debug builds
#if !defined(NDEBUG)
  LOG(debug2) << "EncoderProperties: {"
              << "sourceTypeB: " << sizeof(source_T) << ", "
              << "streamTypeB: " << sizeof(stream_T) << ", "
              << "coderTypeB: " << sizeof(coder_T) << ", "
              << "probabilityBits: " << mProbabilityBits << ", "
              << "inputBufferSizeB: " << inputBufferSize * sizeof(source_T) << ", "
              << "outputBufferSizeB: " << std::distance(outputBegin, outputIter) * sizeof(stream_T) << "}";
#endif

  LOG(trace) << "done encoding";

  return outputIter;
};

} // namespace rans
} // namespace o2

#endif /* RANS_ENCODER_H */
