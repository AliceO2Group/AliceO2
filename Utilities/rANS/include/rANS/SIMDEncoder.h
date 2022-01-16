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

#ifndef RANS_SIMDENCODER_H
#define RANS_SIMDENCODER_H

#include <memory>
#include <algorithm>
#include <iomanip>

#include <fairlogger/Logger.h>
#include <stdexcept>

#include "rANS/internal/backend/simd/types.h"
#include "rANS/internal/backend/simd/Encoder.h"
#include "rANS/internal/backend/simd/Symbol.h"
#include "rANS/internal/helper.h"
#include "rANS/RenormedFrequencyTable.h"
#include "rANS/internal/backend/simd/SymbolTable.h"

namespace o2
{
namespace rans
{

namespace internal
{

// template <typename source_IT>
// inline source_IT getSymbols(source_IT symbolIter, const internal::simd::SymbolTable& symbolTable, const simd::Symbol** dst) noexcept
// {
//   const source_IT ptr = symbolIter - 4;
//   const symbol_t minSymbol = symbolTable.getMinSymbol();
//   const auto tableBegin = symbolTable.data();

//   if constexpr (std::is_pointer_v<source_IT> && std::is_same_v<typename std::iterator_traits<source_IT>::value_type, uint32_t>) {
//     const uintptr_t diff = reinterpret_cast<uintptr_t>(tableBegin - minSymbol);
//     __m128i sourceData = _mm_loadu_si128(reinterpret_cast<const __m128i_u*>(ptr));
//     __m256i index = _mm256_cvtepi32_epi64(sourceData);
//     static_assert(sizeof(simd::Symbol) == (1u << 4u));
//     index = _mm256_slli_epi64(index, 4);
//     index = _mm256_add_epi64(index, _mm256_set1_epi64x(diff));

//     _mm256_store_si256(reinterpret_cast<__m256i*>(dst), index);

//     auto pos = [&](size_t i) { return reinterpret_cast<uintptr_t>(tableBegin + *(symbolIter - i) - minSymbol); };

//   } else {
//     dst[0] = tableBegin + *(symbolIter - 4) - minSymbol;
//     dst[1] = tableBegin + *(symbolIter - 3) - minSymbol;
//     dst[2] = tableBegin + *(symbolIter - 2) - minSymbol;
//     dst[3] = tableBegin + *(symbolIter - 1) - minSymbol;
//   }
//   return ptr;
// };

template <typename source_IT, uint8_t nHardwareStreams_V>
inline std::tuple<source_IT, simd::UnrolledSymbols> getSymbols(source_IT symbolIter, const simd::SymbolTable& symbolTable) noexcept
{
  const symbol_t minSymbol = symbolTable.getMinSymbol();
  const auto tableBegin = symbolTable.data();

  simd::UnrolledSymbols unrolledSymbols;

  if constexpr (nHardwareStreams_V == 4) {
    using namespace simd;

    AlignedArray<const Symbol*, SIMDWidth::SSE, 4> ret{tableBegin + *(symbolIter - 4) - minSymbol,
                                                       tableBegin + *(symbolIter - 3) - minSymbol,
                                                       tableBegin + *(symbolIter - 2) - minSymbol,
                                                       tableBegin + *(symbolIter - 1) - minSymbol};

    aosToSoa(ArrayView{ret}.template subView<0, 2>(), toSIMDView(unrolledSymbols.frequencies).template subView<0, 1>(), toSIMDView(unrolledSymbols.cumulativeFrequencies).template subView<0, 1>());
    aosToSoa(ArrayView{ret}.template subView<2, 2>(), toSIMDView(unrolledSymbols.frequencies).template subView<1, 1>(), toSIMDView(unrolledSymbols.cumulativeFrequencies).template subView<1, 1>());

    return {symbolIter - nHardwareStreams_V, unrolledSymbols};
  } else {
    using namespace simd;
    AlignedArray<const Symbol*, SIMDWidth::SSE, 8> ret{tableBegin + *(symbolIter - 8) - minSymbol,
                                                       tableBegin + *(symbolIter - 7) - minSymbol,
                                                       tableBegin + *(symbolIter - 6) - minSymbol,
                                                       tableBegin + *(symbolIter - 5) - minSymbol,
                                                       tableBegin + *(symbolIter - 4) - minSymbol,
                                                       tableBegin + *(symbolIter - 3) - minSymbol,
                                                       tableBegin + *(symbolIter - 2) - minSymbol,
                                                       tableBegin + *(symbolIter - 1) - minSymbol};
    aosToSoa(ArrayView{ret}.template subView<0, 4>(), toSIMDView(unrolledSymbols.frequencies).template subView<0, 1>(), toSIMDView(unrolledSymbols.cumulativeFrequencies).template subView<0, 1>());
    aosToSoa(ArrayView{ret}.template subView<4, 4>(), toSIMDView(unrolledSymbols.frequencies).template subView<1, 1>(), toSIMDView(unrolledSymbols.cumulativeFrequencies).template subView<1, 1>());
    return {symbolIter - nHardwareStreams_V, unrolledSymbols};
  }
};
} // namespace internal

template <typename coder_T,
          typename stream_T,
          typename source_T,
          uint8_t nStreams_V = 4,
          uint8_t nHardwareStreams_V = 2>
class SIMDEncoder
{
 protected:
  using encoderSymbolTable_t = internal::simd::SymbolTable;

 public:
  //TODO(milettri): fix once ROOT cling respects the standard http://wg21.link/p1286r2
  SIMDEncoder() noexcept {}; //NOLINT
  SIMDEncoder(const RenormedFrequencyTable& frequencyTable) : mSymbolTable{frequencyTable} {};

  inline size_t getSymbolTablePrecision() const noexcept { return mSymbolTable.getPrecision(); };
  inline size_t getAlphabetRangeBits() const noexcept { return mSymbolTable.getAlphabetRangeBits(); };
  inline symbol_t getMinSymbol() const noexcept { return mSymbolTable.getMinSymbol(); };
  inline symbol_t getMaxSymbol() const noexcept { return mSymbolTable.getMaxSymbol(); };

  template <typename stream_IT, typename source_IT, std::enable_if_t<internal::isCompatibleIter_v<source_T, source_IT>, bool> = true>
  stream_IT process(source_IT inputBegin, source_IT inputEnd, stream_IT outputBegin) const;

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
stream_IT SIMDEncoder<coder_T, stream_T, source_T, nStreams_V, nHardwareStreams_V>::process(source_IT inputBegin, source_IT inputEnd, stream_IT outputBegin) const
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

  auto maskedEncode = [this](source_IT symbolIter, stream_IT outputIter, ransCoder_t& coder, size_t nActiveStreams = nParallelStreams_V) {
    std::array<const internal::simd::Symbol*, nParallelStreams_V> encoderSymbols{};
    for (auto encSymbolIter = encoderSymbols.rend() - nActiveStreams; encSymbolIter != encoderSymbols.rend(); ++encSymbolIter) {
      const source_T symbol = *(--symbolIter);
      *encSymbolIter = &(this->mSymbolTable)[symbol];
    }
    return std::tuple(symbolIter, coder.putSymbols(outputIter, encoderSymbols, nActiveStreams));
  };

  auto encode = [this](source_IT symbolIter, stream_IT outputIter, ransCoder_t& coder) {
    auto [newSymbolIter, encoderSymbols] = getSymbols<source_IT, nParallelStreams_V>(symbolIter, this->mSymbolTable);
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

#endif /* RANS_SIMDENCODER_H */
