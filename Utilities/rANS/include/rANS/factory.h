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

/// @file   factory.h
/// @author michael.lettrich@cern.ch
/// @brief  static factory classes for building histograms, encoders and decoders.

#ifndef RANS_FACTORY_H_
#define RANS_FACTORY_H_

#ifdef __CLING__
#error rANS should not be exposed to root
#endif

#include "rANS/internal/common/defaults.h"
#include "rANS/internal/common/typetraits.h"
#include "rANS/internal/common/codertraits.h"

#include "rANS/internal/metrics/Metrics.h"
#include "rANS/internal/transform/renorm.h"

#include "rANS/internal/containers/DenseHistogram.h"
#include "rANS/internal/containers/DenseSymbolTable.h"

#include "rANS/internal/containers/AdaptiveHistogram.h"
#include "rANS/internal/containers/AdaptiveSymbolTable.h"

#include "rANS/internal/containers/SparseSymbolTable.h"
#include "rANS/internal/containers/SparseHistogram.h"

#include "rANS/internal/containers/RenormedHistogram.h"

#include "rANS/internal/containers/LowRangeDecoderTable.h"
#include "rANS/internal/containers/HighRangeDecoderTable.h"
#include "rANS/internal/containers/Symbol.h"

#include "rANS/internal/encode/Encoder.h"
#include "rANS/internal/encode/SingleStreamEncoderImpl.h"
#include "rANS/internal/encode/SIMDEncoderImpl.h"

#include "rANS/internal/decode/Decoder.h"
#include "rANS/internal/decode/DecoderImpl.h"

namespace o2::rans
{

namespace internal
{

template <template <class... types> class histogram_T>
struct makeHistogram {

  template <typename source_T>
  using histogram_type = histogram_T<source_T>;

  template <typename source_IT>
  [[nodiscard]] inline static decltype(auto) fromSamples(source_IT begin, source_IT end)
  {
    using source_type = typename std::iterator_traits<source_IT>::value_type;

    histogram_type<source_type> f{};
    f.addSamples(begin, end);
    return f;
  };

  template <typename source_T>
  [[nodiscard]] inline static decltype(auto) fromSamples(gsl::span<const source_T> range)
  {
    using source_type = typename std::remove_cv_t<source_T>;

    histogram_type<source_type> f;
    f.addSamples(range);
    return f;
  };
};

template <template <typename source_T, typename symbol_T> class symbolTable_T,
          CoderTag coderTag_V = defaults::DefaultTag,
          size_t nStreams_V = defaults::CoderPreset<coderTag_V>::nStreams,
          size_t renormingLowerBound_V = defaults::CoderPreset<coderTag_V>::renormingLowerBound>
class makeEncoder
{
 private:
  static constexpr size_t NStreams = nStreams_V;
  static constexpr size_t RenormingLowerBound = renormingLowerBound_V;
  static constexpr CoderTag coderTag = coderTag_V;

  using this_type = makeEncoder<symbolTable_T, coderTag_V, nStreams_V, renormingLowerBound_V>;
  using symbol_type = typename internal::SymbolTraits<coderTag>::type;
  using coder_command = typename internal::CoderTraits<coderTag>::template type<this_type::RenormingLowerBound>;
  template <typename source_T>
  using symbolTable_type = symbolTable_T<source_T, symbol_type>;
  template <typename source_T>
  using encoderType = Encoder<coder_command, symbolTable_type<source_T>, this_type::NStreams>;

 public:
  template <typename container_T>
  [[nodiscard]] inline static constexpr decltype(auto) fromRenormed(const RenormedHistogramConcept<container_T>& renormed)
  {
    using source_type = typename RenormedHistogramConcept<container_T>::source_type;
    return encoderType<source_type>{renormed};
  };

  template <typename histogram_T>
  [[nodiscard]] inline static decltype(auto) fromHistogram(histogram_T histogram, RenormingPolicy renormingPolicy = RenormingPolicy::Auto)
  {
    static_assert(internal::isHistogram_v<histogram_T>);
    const auto renormedHistogram = renorm(std::move(histogram), renormingPolicy);
    return this_type::fromRenormed(renormedHistogram);
  };

  template <typename histogram_T>
  [[nodiscard]] inline static decltype(auto) fromHistogram(histogram_T histogram, Metrics<typename histogram_T::source_type>& metrics, RenormingPolicy renormingPolicy = RenormingPolicy::Auto)
  {
    static_assert(internal::isHistogram_v<histogram_T>);
    const auto renormedHistogram = renorm(std::move(histogram), metrics, renormingPolicy);
    return this_type::fromRenormed(renormedHistogram);
  };

  template <typename histogram_T>
  [[nodiscard]] inline static decltype(auto) fromHistogram(histogram_T histogram, size_t renormingPrecision, RenormingPolicy renormingPolicy = RenormingPolicy::Auto)
  {
    static_assert(internal::isHistogram_v<histogram_T>);
    const auto renormedHistogram = renorm(std::move(histogram), renormingPrecision, renormingPolicy);
    return this_type::fromRenormed(renormedHistogram);
  };
};
} // namespace internal

struct makeDenseHistogram : public internal::makeHistogram<DenseHistogram> {
  using base_type = internal::makeHistogram<DenseHistogram>;

  using base_type::fromSamples;

  template <typename source_IT>
  [[nodiscard]] inline static decltype(auto) fromSamples(source_IT begin, source_IT end,
                                                         typename std::iterator_traits<source_IT>::value_type min,
                                                         typename std::iterator_traits<source_IT>::value_type max)
  {
    using source_type = typename std::iterator_traits<source_IT>::value_type;
    using histogram_type = DenseHistogram<source_type>;

    histogram_type f{};
    f.addSamples(begin, end, min, max);
    return f;
  };

  template <typename source_T>
  [[nodiscard]] inline static decltype(auto) fromSamples(gsl::span<const source_T> range, source_T min, source_T max)
  {
    using source_type = typename std::remove_cv_t<source_T>;
    using histogram_type = DenseHistogram<source_type>;

    histogram_type f;
    f.addSamples(range, min, max);
    return f;
  };
};

using makeAdaptiveHistogram = internal::makeHistogram<AdaptiveHistogram>;

using makeSparseHistogram = internal::makeHistogram<SparseHistogram>;

template <CoderTag coderTag_V = defaults::DefaultTag,
          size_t nStreams_V = defaults::CoderPreset<coderTag_V>::nStreams,
          size_t renormingLowerBound_V = defaults::CoderPreset<coderTag_V>::renormingLowerBound>
using makeDenseEncoder = internal::makeEncoder<DenseSymbolTable, coderTag_V, nStreams_V, renormingLowerBound_V>;

template <CoderTag coderTag_V = defaults::DefaultTag,
          size_t nStreams_V = defaults::CoderPreset<coderTag_V>::nStreams,
          size_t renormingLowerBound_V = defaults::CoderPreset<coderTag_V>::renormingLowerBound>
using makeAdaptiveEncoder = internal::makeEncoder<AdaptiveSymbolTable, coderTag_V, nStreams_V, renormingLowerBound_V>;

template <CoderTag coderTag_V = defaults::DefaultTag,
          size_t nStreams_V = defaults::CoderPreset<coderTag_V>::nStreams,
          size_t renormingLowerBound_V = defaults::CoderPreset<coderTag_V>::renormingLowerBound>
using makeSparseEncoder = internal::makeEncoder<SparseSymbolTable, coderTag_V, nStreams_V, renormingLowerBound_V>;

template <size_t renormingLowerBound_V = defaults::internal::RenormingLowerBound>
class makeDecoder
{

  using this_type = makeDecoder<renormingLowerBound_V>;

 public:
  template <typename source_T>
  [[nodiscard]] inline static constexpr decltype(auto) fromRenormed(const RenormedDenseHistogram<source_T>& renormed)
  {
    using namespace internal;

    using source_type = source_T;
    using coder_type = DecoderImpl<renormingLowerBound_V>;
    using decoder_type = Decoder<source_type, coder_type>;

    return decoder_type{renormed};
  };

  template <typename source_T>
  [[nodiscard]] inline static decltype(auto) fromHistogram(DenseHistogram<source_T> histogram, RenormingPolicy renormingPolicy = RenormingPolicy::Auto)
  {
    const auto renormedHistogram = renorm(std::move(histogram), renormingPolicy);
    return this_type::fromRenormed(renormedHistogram);
  };

  template <typename source_T>
  [[nodiscard]] inline static decltype(auto) fromHistogram(DenseHistogram<source_T> histogram, Metrics<source_T>& metrics, RenormingPolicy renormingPolicy = RenormingPolicy::Auto)
  {
    const auto renormedHistogram = renorm(std::move(histogram), metrics, renormingPolicy);
    return this_type::fromRenormed(renormedHistogram);
  };

  template <typename source_T>
  [[nodiscard]] inline static decltype(auto) fromHistogram(DenseHistogram<source_T> histogram, size_t renormingPrecision, RenormingPolicy renormingPolicy = RenormingPolicy::Auto)
  {
    const auto renormedHistogram = renorm(std::move(histogram), renormingPrecision);
    return this_type::fromRenormed(renormedHistogram);
  };
};

template <typename source_T>
using denseEncoder_type = decltype(makeDenseEncoder<>::fromRenormed(RenormedDenseHistogram<source_T>{}));

template <typename source_T>
using adaptiveEncoder_type = decltype(makeAdaptiveEncoder<>::fromRenormed(RenormedAdaptiveHistogram<source_T>{}));

template <typename source_T>
using sparseEncoder_type = decltype(makeSparseEncoder<>::fromRenormed(RenormedSparseHistogram<source_T>{}));

template <typename source_T>
using defaultDecoder_type = decltype(makeDecoder<>::fromRenormed(RenormedDenseHistogram<source_T>{}));

} // namespace o2::rans

#endif /* RANS_FACTORY_H_ */