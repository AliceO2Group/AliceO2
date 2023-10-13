// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   compat.h
/// @author Michael Lettrich
/// @brief  functionality to maintain compatibility with previous version of this library

#ifndef RANS_COMPAT_H_
#define RANS_COMPAT_H_

#ifdef __CLING__
#error rANS should not be exposed to root
#endif

#include <numeric>

#include "rANS/internal/common/typetraits.h"

#include "rANS/internal/containers/DenseHistogram.h"
#include "rANS/internal/containers/RenormedHistogram.h"
#include "rANS/internal/containers/DenseSymbolTable.h"
#include "rANS/internal/containers/Symbol.h"
#include "rANS/internal/containers/LowRangeDecoderTable.h"

#include "rANS/internal/encode/Encoder.h"
#include "rANS/internal/encode/SingleStreamEncoderImpl.h"

#include "rANS/internal/decode/Decoder.h"
#include "rANS/internal/decode/DecoderImpl.h"

#include "rANS/factory.h"

namespace o2::rans::compat
{

namespace defaults
{
namespace internal
{
inline constexpr size_t RenormingLowerBound = 31;
} // namespace internal

struct CoderPreset {
  inline static constexpr size_t nStreams = 2;
  inline static constexpr size_t renormingLowerBound = internal::RenormingLowerBound;
};

} // namespace defaults

namespace compatImpl
{
inline constexpr uint32_t MinRenormThreshold = 10;
inline constexpr uint32_t MaxRenormThreshold = 20;
} // namespace compatImpl

inline size_t computeRenormingPrecision(size_t nUsedAlphabetSymbols)
{
  const uint32_t minBits = o2::rans::utils::log2UInt(nUsedAlphabetSymbols);
  const uint32_t estimate = minBits * 3u / 2u;
  const uint32_t maxThreshold = std::max(minBits, compatImpl::MaxRenormThreshold);
  const uint32_t minThreshold = std::max(estimate, compatImpl::MinRenormThreshold);

  return std::min(minThreshold, maxThreshold);
};

template <typename source_T>
RenormedDenseHistogram<source_T> renorm(DenseHistogram<source_T> histogram, size_t newPrecision = 0)
{
  using namespace o2::rans::internal;

  constexpr size_t IncompressibleSymbolFrequency = 1;

  if (histogram.empty()) {
    LOG(warning) << "rescaling empty histogram";
  }

  size_t nUsedAlphabetSymbols = countNUsedAlphabetSymbols(histogram);

  if (newPrecision == 0) {
    newPrecision = computeRenormingPrecision(nUsedAlphabetSymbols);
  }

  const size_t alphabetSize = histogram.size() + 1;              // +1 for incompressible symbol
  std::vector<uint64_t> cumulativeFrequencies(alphabetSize + 1); // +1 to store total cumulative frequencies
  cumulativeFrequencies[0] = 0;
  std::inclusive_scan(histogram.begin(), histogram.end(), ++cumulativeFrequencies.begin(), std::plus<>(), 0ull);
  cumulativeFrequencies.back() = histogram.getNumSamples() + IncompressibleSymbolFrequency;

  auto getFrequency = [&cumulativeFrequencies](count_t i) {
    assert(cumulativeFrequencies[i + 1] >= cumulativeFrequencies[i]);
    return cumulativeFrequencies[i + 1] - cumulativeFrequencies[i];
  };

  // we will memorize only those entries which can be used
  const auto sortIdx = [&]() {
    std::vector<size_t> indices;
    indices.reserve(nUsedAlphabetSymbols + 1);

    for (size_t i = 0; i < alphabetSize; ++i) {
      if (getFrequency(i) != 0) {
        indices.push_back(i);
      }
    }

    std::sort(indices.begin(), indices.end(), [&](count_t i, count_t j) { return getFrequency(i) < getFrequency(j); });
    return indices;
  }();

  // resample distribution based on cumulative frequencies
  const count_t newCumulatedFrequency = utils::pow2(newPrecision);
  size_t nSamples = histogram.getNumSamples() + IncompressibleSymbolFrequency;
  assert(newCumulatedFrequency >= nUsedAlphabetSymbols);
  size_t needsShift = 0;
  for (size_t i = 0; i < sortIdx.size(); i++) {
    if (static_cast<count_t>(getFrequency(sortIdx[i])) * (newCumulatedFrequency - needsShift) / nSamples >= 1) {
      break;
    }
    needsShift++;
  }

  size_t shift = 0;
  auto beforeUpdate = cumulativeFrequencies[0];
  for (size_t i = 0; i < alphabetSize; i++) {
    auto& nextCumulative = cumulativeFrequencies[i + 1];
    uint64_t oldFrequeny = nextCumulative - beforeUpdate;
    if (oldFrequeny && oldFrequeny * (newCumulatedFrequency - needsShift) / nSamples < 1) {
      shift++;
    }
    beforeUpdate = cumulativeFrequencies[i + 1];
    nextCumulative = (static_cast<uint64_t>(newCumulatedFrequency - needsShift) * nextCumulative) / nSamples + shift;
  }
  assert(shift == needsShift);

  // verify
#if !defined(NDEBUG)
  assert(cumulativeFrequencies.front() == 0);
  assert(cumulativeFrequencies.back() == newCumulatedFrequency);
  size_t i = 0;
  for (auto frequency : histogram) {
    if (frequency == 0) {
      assert(cumulativeFrequencies[i + 1] == cumulativeFrequencies[i]);
    } else {
      assert(cumulativeFrequencies[i + 1] > cumulativeFrequencies[i]);
    }
    ++i;
  }
#endif

  typename RenormedDenseHistogram<source_T>::container_type rescaledFrequencies(histogram.size(), histogram.getOffset());

  assert(cumulativeFrequencies.size() == histogram.size() + 2);
  // calculate updated frequencies
  for (size_t i = 0; i < histogram.size(); ++i) {
    rescaledFrequencies(i) = getFrequency(i);
  }
  const typename RenormedDenseHistogram<source_T>::value_type incompressibleSymbolFrequency = getFrequency(histogram.size());

  return RenormedDenseHistogram<source_T>{std::move(rescaledFrequencies), newPrecision, incompressibleSymbolFrequency};
};

class makeEncoder
{

 public:
  template <typename container_T>
  [[nodiscard]] inline static constexpr decltype(auto) fromRenormed(const RenormedHistogramConcept<container_T>& renormed)
  {
    using namespace o2::rans::internal;
    using source_type = typename RenormedHistogramConcept<container_T>::source_type;
    using symbol_type = internal::PrecomputedSymbol;
    using coder_command = SingleStreamEncoderImpl<mRenormingLowerBound>;
    using symbolTable_type = DenseSymbolTable<source_type, symbol_type>;
    using encoderType = Encoder<coder_command, symbolTable_type, mNstreams>;

    return encoderType{renormed};
  };

  template <typename source_T>
  [[nodiscard]] inline static decltype(auto) fromHistogram(DenseHistogram<source_T> histogram, size_t renormingPrecision = 0)
  {
    const auto renormedHistogram = o2::rans::compat::renorm(std::move(histogram), renormingPrecision);
    return makeEncoder::fromRenormed(renormedHistogram);
  };

  template <typename source_IT>
  [[nodiscard]] inline static decltype(auto) fromSamples(source_IT begin, source_IT end, size_t renormingPrecision = 0)
  {
    auto histogram = makeDenseHistogram::fromSamples(begin, end);

    return makeEncoder::fromHistogram(std::move(histogram), renormingPrecision);
  };

  template <typename source_T>
  [[nodiscard]] inline static decltype(auto) fromSamples(gsl::span<const source_T> range, size_t renormingPrecision = 0)
  {
    auto histogram = makeDenseHistogram::template fromSamples(range);
    return makeEncoder::fromHistogram(std::move(histogram), renormingPrecision);
  };

 private:
  static constexpr CoderTag mCoderTag = CoderTag::SingleStream;
  static constexpr size_t mNstreams = defaults::CoderPreset::nStreams;
  static constexpr size_t mRenormingLowerBound = defaults::CoderPreset::renormingLowerBound;
};

class makeDecoder
{

  using this_type = makeDecoder;

 public:
  template <typename container_T>
  [[nodiscard]] inline static constexpr decltype(auto) fromRenormed(const RenormedHistogramConcept<container_T>& renormed)
  {
    using namespace internal;

    using source_type = typename RenormedHistogramConcept<container_T>::source_type;
    using coder_type = DecoderImpl<mRenormingLowerBound>;
    using decoder_type = Decoder<source_type, coder_type>;

    return decoder_type{renormed};
  };

  template <typename source_T>
  [[nodiscard]] inline static decltype(auto) fromHistogram(DenseHistogram<source_T> histogram, size_t renormingPrecision = 0)
  {
    const auto renormedHistogram = o2::rans::compat::renorm(std::move(histogram), renormingPrecision);
    return this_type::fromRenormed(renormedHistogram);
  };

  template <typename source_IT>
  [[nodiscard]] inline static decltype(auto) fromSamples(source_IT begin, source_IT end, size_t renormingPrecision = 0)
  {
    auto histogram = makeDenseHistogram::fromSamples(begin, end);
    return this_type::fromHistogram(std::move(histogram), renormingPrecision);
  };

  template <typename source_T>
  [[nodiscard]] inline static decltype(auto) fromSamples(gsl::span<const source_T> range, size_t renormingPrecision = 0)
  {
    auto histogram = makeDenseHistogram::fromSamples(range);
    return this_type::fromHistogram(std::move(histogram), renormingPrecision);
  };

 private:
  static constexpr CoderTag mCoderTag = CoderTag::SingleStream;
  static constexpr size_t mNstreams = defaults::CoderPreset::nStreams;
  static constexpr size_t mRenormingLowerBound = defaults::CoderPreset::renormingLowerBound;
};

template <typename source_T>
inline size_t getAlphabetRangeBits(const DenseHistogram<source_T>& histogram) noexcept
{
  using namespace o2::rans::internal;
  const auto view = trim(makeHistogramView(histogram));
  return internal::numBitsForNSymbols(view.size());
};

template <typename source_T>
inline size_t getAlphabetRangeBits(const RenormedDenseHistogram<source_T>& histogram) noexcept
{
  using namespace o2::rans::internal;
  const auto view = trim(makeHistogramView(histogram));
  return internal::numBitsForNSymbols(view.size() + histogram.hasIncompressibleSymbol());
};

template <typename source_T, typename symbol_T>
inline size_t getAlphabetRangeBits(const DenseSymbolTable<source_T, symbol_T>& symbolTable) noexcept
{
  const bool hasIncompressibleSymbol = symbolTable.getEscapeSymbol().getFrequency() > 0;
  return internal::numBitsForNSymbols(symbolTable.size() + hasIncompressibleSymbol);
};

inline size_t calculateMaxBufferSizeB(size_t nElements, size_t rangeBits)
{
  constexpr size_t sizeofStreamT = sizeof(uint32_t);
  //  // RS: w/o safety margin the o2-test-ctf-io produces an overflow in the Encoder::process
  //  constexpr size_t SaferyMargin = 16;
  //  return std::ceil(1.20 * (num * rangeBits * 1.0) / (sizeofStreamT * 8.0)) + SaferyMargin;
  return nElements * sizeofStreamT;
}

template <typename source_T>
using encoder_type = decltype(makeEncoder::fromRenormed(RenormedDenseHistogram<source_T>{}));

template <typename source_T>
using decoder_type = decltype(makeDecoder::fromRenormed(RenormedDenseHistogram<source_T>{}));

} // namespace o2::rans::compat

#endif /* RANS_COMPAT_H_ */