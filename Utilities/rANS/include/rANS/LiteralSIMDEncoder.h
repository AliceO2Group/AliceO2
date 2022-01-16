// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   SIMDEncoder.h
/// @author Michael Lettrich
/// @since  2021-03-18
/// @brief  Encoder - code symbol into a rANS encoded state

#ifndef RANS_LITERALSIMDENCODER_H
#define RANS_LITERALSIMDENCODER_H

#include <memory>
#include <algorithm>
#include <iomanip>

#include <fairlogger/Logger.h>
#include <stdexcept>

#include "rANS/internal/backend/simd/types.h"
#include "rANS/internal/backend/simd/Encoder.h"
#include "rANS/internal/backend/simd/Symbol.h"
#include "rANS/internal/backend/simd/SymbolMapper.h"
#include "rANS/internal/helper.h"
#include "rANS/RenormedFrequencyTable.h"
#include "rANS/internal/backend/simd/SymbolTable.h"
#include "rANS/SIMDEncoder.h"

namespace o2
{
namespace rans
{

namespace internal
{
template <typename source_IT>
inline const internal::simd::Symbol* lookupSymbol(source_IT iter, const simd::SymbolTable& symbolTable, std::vector<typename std::iterator_traits<source_IT>::value_type>& literals) noexcept
{
  const auto symbol = *iter;
  const auto* encoderSymbol = &(symbolTable[symbol]);
  if (symbolTable.isEscapeSymbol(*encoderSymbol)) {
    literals.push_back(symbol);
  }
  return encoderSymbol;
};

template <typename source_IT, uint8_t nHardwareStreams_V>
inline std::tuple<source_IT, simd::AlignedArray<const internal::simd::Symbol*, simd::SIMDWidth::AVX, nHardwareStreams_V>> getSymbols(source_IT symbolIter, const simd::SymbolTable& symbolTable, std::vector<typename std::iterator_traits<source_IT>::value_type>& literals) noexcept
{
  using return_t = simd::AlignedArray<const internal::simd::Symbol*, simd::SIMDWidth::AVX, nHardwareStreams_V>;

  if constexpr (nHardwareStreams_V == 2) {
    return_t ret;
    ret[1] = lookupSymbol(symbolIter - 1, symbolTable, literals);
    ret[0] = lookupSymbol(symbolIter - 2, symbolTable, literals);

    return {symbolIter - 2, ret};

    // return {
    //   symbolIter - 2,
    //   {lookupSymbol(symbolIter - 2, symbolTable, literals),
    //    lookupSymbol(symbolIter - 1, symbolTable, literals)}};
  } else if constexpr (nHardwareStreams_V == 4) {
    // return {
    //   symbolIter - 4,
    //   {
    //     lookupSymbol(symbolIter - 4, symbolTable, literals),
    //     lookupSymbol(symbolIter - 3, symbolTable, literals),
    //     lookupSymbol(symbolIter - 2, symbolTable, literals),
    //     lookupSymbol(symbolIter - 1, symbolTable, literals),
    //   }};
    return_t ret;
    ret[3] = lookupSymbol(symbolIter - 1, symbolTable, literals);
    ret[2] = lookupSymbol(symbolIter - 2, symbolTable, literals);
    ret[1] = lookupSymbol(symbolIter - 3, symbolTable, literals);
    ret[0] = lookupSymbol(symbolIter - 4, symbolTable, literals);

    return {symbolIter - 4, ret};

  } else if constexpr (nHardwareStreams_V == 8) {
    // return {
    //   symbolIter - 8,
    //   {lookupSymbol(symbolIter - 8, symbolTable, literals),
    //    lookupSymbol(symbolIter - 7, symbolTable, literals),
    //    lookupSymbol(symbolIter - 6, symbolTable, literals),
    //    lookupSymbol(symbolIter - 5, symbolTable, literals),
    //    lookupSymbol(symbolIter - 4, symbolTable, literals),
    //    lookupSymbol(symbolIter - 3, symbolTable, literals),
    //    lookupSymbol(symbolIter - 2, symbolTable, literals),
    //    lookupSymbol(symbolIter - 1, symbolTable, literals)}};

    return_t ret;
    ret[7] = lookupSymbol(symbolIter - 1, symbolTable, literals);
    ret[6] = lookupSymbol(symbolIter - 2, symbolTable, literals);
    ret[5] = lookupSymbol(symbolIter - 3, symbolTable, literals);
    ret[4] = lookupSymbol(symbolIter - 4, symbolTable, literals);
    ret[3] = lookupSymbol(symbolIter - 5, symbolTable, literals);
    ret[2] = lookupSymbol(symbolIter - 6, symbolTable, literals);
    ret[1] = lookupSymbol(symbolIter - 7, symbolTable, literals);
    ret[0] = lookupSymbol(symbolIter - 8, symbolTable, literals);

    return {symbolIter - 8, ret};
  }
};
} // namespace internal

template <typename coder_T,
          typename stream_T,
          typename source_T,
          uint8_t nStreams_V = 4,
          uint8_t nHardwareStreams_V = 2>
class LiteralSIMDEncoder
{
 protected:
  using encoderSymbolTable_t = internal::simd::SymbolTable;

 public:
  using source_t = source_T;

  //TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  LiteralSIMDEncoder() noexcept {}; //NOLINT
  LiteralSIMDEncoder(const RenormedFrequencyTable& frequencyTable) : mSymbolTable{frequencyTable} {};

  inline size_t getSymbolTablePrecision() const noexcept { return mSymbolTable.getPrecision(); };
  inline size_t getAlphabetRangeBits() const noexcept { return mSymbolTable.getAlphabetRangeBits(); };
  inline symbol_t getMinSymbol() const noexcept { return mSymbolTable.getMinSymbol(); };
  inline symbol_t getMaxSymbol() const noexcept { return mSymbolTable.getMaxSymbol(); };

  template <typename stream_IT, typename source_IT, std::enable_if_t<internal::isCompatibleIter_v<source_T, source_IT>, bool> = true>
  stream_IT process(source_IT inputBegin, source_IT inputEnd, stream_IT outputBegin, std::vector<source_T>& literals) const;

 protected:
  encoderSymbolTable_t mSymbolTable{};
  size_t mSymbolTablePrecision{};

  //TODO(milettri): make this depend on hardware
  static constexpr size_t nParallelStreams_V = nHardwareStreams_V * 2;

  static_assert(nStreams_V >= nParallelStreams_V);
  static_assert(nStreams_V % nParallelStreams_V == 0);

  static constexpr size_t nInterleavedStreams_V = nStreams_V / nParallelStreams_V;
  static constexpr internal::simd::SIMDWidth simdWidth = internal::simd::getSimdWidth<coder_T>(nHardwareStreams_V);
  using ransCoder_t = typename internal::simd::Encoder<simdWidth>;
};

template <typename coder_T, typename stream_T, typename source_T, uint8_t nStreams_V, uint8_t nHardwareStreams_V>
template <typename stream_IT,
          typename source_IT,
          std::enable_if_t<internal::isCompatibleIter_v<source_T, source_IT>, bool>>
stream_IT LiteralSIMDEncoder<coder_T, stream_T, source_T, nStreams_V, nHardwareStreams_V>::process(source_IT inputBegin, source_IT inputEnd, stream_IT outputBegin, std::vector<source_T>& literals) const
{
  using namespace internal;
  // LOG(trace) << "start encoding";
  RANSTimer t;
  t.start();

  if (inputBegin == inputEnd) {
    LOG(warning) << "passed empty message to encoder, skip encoding";
    return outputBegin;
  }

  stream_IT outputIter = outputBegin;
  source_IT inputIT = inputEnd;

  simd::SymbolMapper<simdWidth> mapper{this->mSymbolTable};
  auto literalIter = literals.data();

  auto maskedEncode = [this, &literals](source_IT symbolIter, stream_IT outputIter, ransCoder_t& coder, size_t nActiveStreams = nParallelStreams_V) {
    std::array<const internal::simd::Symbol*, nParallelStreams_V> encoderSymbols{};
    for (auto encSymbolIter = encoderSymbols.rend() - nActiveStreams; encSymbolIter != encoderSymbols.rend(); ++encSymbolIter) {
      *encSymbolIter = o2::rans::internal::lookupSymbol(--symbolIter, this->mSymbolTable, literals);
    }
    return std::tuple(symbolIter, coder.putSymbols(outputIter, encoderSymbols, nActiveStreams));
  };

  auto encode = [this, &literalIter, &literals, &mapper](source_IT symbolIter, stream_IT outputIter, ransCoder_t& coder) {
    auto [newSymbolIter, newLiteralIter, encoderSymbols] = mapper.template readSymbols<source_IT>(symbolIter, literalIter);
    literalIter = newLiteralIter;
    return std::make_tuple(newSymbolIter, coder.putSymbols(outputIter, encoderSymbols));
  };

  // create coders
  std::array<ransCoder_t, nInterleavedStreams_V> interleavedCoders;
  for (auto& coder : interleavedCoders) {
    coder = ransCoder_t(this->getSymbolTablePrecision());
  }

  // calculate sizes and numbers of iterations:
  const auto inputBufferSize = std::distance(inputBegin, inputEnd);
  const size_t nMainLoopIterations = inputBufferSize / nStreams_V;
  const size_t nMainLoopRemainderSymbols = inputBufferSize % nStreams_V;
  const size_t nRemainderLoopIterations = nMainLoopRemainderSymbols / nParallelStreams_V;
  const size_t nMaskedEncodes = nMainLoopRemainderSymbols % nParallelStreams_V;

  // LOG(trace) << "InputBufferSize: " << inputBufferSize;
  // LOG(trace) << "Loops Main: " << nMainLoopIterations << ", Loops Remainder: " << nMainLoopRemainderSymbols << ", Masked Encodes :" << nMaskedEncodes;

  // iterator pointing to the active coder
  auto coderIter = std::rend(interleavedCoders) - nRemainderLoopIterations;

  if (nMaskedEncodes) {
    // LOG(trace) << "masked encodes";
    // one more encoding step than nRemainderLoopIterations for masked encoding
    // will not cause out of range
    --coderIter;
    std::tie(inputIT, outputIter) = maskedEncode(inputIT, outputIter, *(coderIter), nMaskedEncodes);
    coderIter++;
  }

  // right spot for iterators;
  size_t s = std::distance(inputBegin, inputEnd);
  size_t newSize = ((s / nStreams_V) + 1) * nStreams_V;
  size_t pos = literals.size();
  literals.resize(newSize);
  literalIter = literals.data() + pos;

  // now encode the rest of the remainder symbols
  // LOG(trace) << "remainder";
  for (coderIter; coderIter != std::rend(interleavedCoders); ++coderIter) {
    std::tie(inputIT, outputIter) = encode(inputIT, outputIter, *coderIter);
  }

  // main encoder loop
  // LOG(trace) << "main loop";
  for (size_t i = 0; i < nMainLoopIterations; ++i) {
    for (size_t interleavedCoderIdx = nInterleavedStreams_V; interleavedCoderIdx-- > 0;) {
      std::tie(inputIT, outputIter) = encode(inputIT, outputIter, interleavedCoders[interleavedCoderIdx]);
    }
    // for (coderIter = std::rbegin(interleavedCoders); coderIter != std::rend(interleavedCoders); ++coderIter) {
    //   std::tie(inputIT, outputIter) = encode(inputIT, outputIter, *coderIter);
    // }
  }

  literals.resize(std::distance(literals.data(), literalIter));

  // LOG(trace) << "flushing";
  for (coderIter = std::rbegin(interleavedCoders); coderIter != std::rend(interleavedCoders); ++coderIter) {
    outputIter = coderIter->flush(outputIter);
  }

  // first iterator past the range so that sizes, distances and iterators work correctly.
  ++outputIter;

  t.stop();
  LOG(debug1) << "Encoder::" << __func__ << " {ProcessedBytes: " << inputBufferSize * sizeof(source_T) << ","
              << " inclusiveTimeMS: " << t.getDurationMS() << ","
              << " BandwidthMiBPS: " << std::fixed << std::setprecision(2) << (inputBufferSize * sizeof(source_T) * 1.0) / (t.getDurationS() * 1.0 * (1 << 20)) << "}";

// advanced diagnostics for debug builds
#if !defined(NDEBUG)

  LOG(debug2) << "EncoderProperties: {"
              << "sourceTypeB: " << sizeof(source_T) << ", "
              << "streamTypeB: " << sizeof(stream_T) << ", "
              << "coderTypeB: " << sizeof(coder_T) << ", "
              << "symbolTablePrecision: " << this->mSymbolTablePrecision << ", "
              << "inputBufferSizeB: " << (inputBufferSize * sizeof(source_T)) << "}";
#endif

  LOG(trace) << "done encoding";

  return outputIter;
};

} // namespace rans
} // namespace o2

#endif /* RANS_LITERALSIMDENCODER_H */
