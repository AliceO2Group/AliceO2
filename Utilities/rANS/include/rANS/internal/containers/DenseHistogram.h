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

/// @file   DenseHistogram.h
/// @author Michael Lettrich
/// @brief  Histogram for source symbols used to estimate symbol probabilities for entropy coding

#ifndef RANS_INTERNAL_CONTAINERS_DENSEHISTOGRAM_H_
#define RANS_INTERNAL_CONTAINERS_DENSEHISTOGRAM_H_

#include <algorithm>
#include <cassert>

#include <gsl/span>

#include <fairlogger/Logger.h>
#include <utility>

#include "rANS/internal/common/utils.h"
#include "rANS/internal/common/exceptions.h"
#include "rANS/internal/containers/HistogramConcept.h"
#include "rANS/internal/containers/Container.h"
#include "rANS/internal/containers/HistogramView.h"

#ifdef RANS_SIMD
#include "rANS/internal/common/simdops.h"
#include "rANS/internal/common/simdtypes.h"

#ifdef RANS_OPENMP
#include <omp.h>
#endif

#endif /* RANS_SIMD */

namespace o2::rans
{

namespace internal
{
namespace histogramImpl
{

template <typename source_T>
inline std::pair<source_T, source_T> minmaxImpl(const source_T* begin, const source_T* end)
{
  const auto [minIter, maxIter] = std::minmax_element(begin, end);
  return {*minIter, *maxIter};
};

#ifdef RANS_SIMD
template <>
inline std::pair<uint32_t, uint32_t> minmaxImpl<uint32_t>(const uint32_t* begin, const uint32_t* end)
{
  return internal::simd::minmax(begin, end);
};

template <>
inline std::pair<int32_t, int32_t> minmaxImpl<int32_t>(const int32_t* begin, const int32_t* end)
{
  return internal::simd::minmax(begin, end);
};
#endif /* RANS_SIMD */

}; // namespace histogramImpl

template <typename source_T>
inline std::pair<source_T, source_T> minmax(gsl::span<const source_T> range)
{
  const auto begin = range.data();
  const auto end = begin + range.size();
  return histogramImpl::minmaxImpl<source_T>(begin, end);
};

} // namespace internal

namespace utils
{
template <typename source_T>
inline std::pair<source_T, source_T> minmax(gsl::span<const source_T> range)
{
  return internal::minmax(range);
}
} // namespace utils

template <typename source_T, typename = void>
class DenseHistogram;

template <typename source_T>
class DenseHistogram<source_T, std::enable_if_t<sizeof(source_T) == 4>> : public internal::VectorContainer<source_T, uint32_t>,
                                                                          public internal::HistogramConcept<source_T,
                                                                                                            typename internal::VectorContainer<source_T, uint32_t>::value_type,
                                                                                                            typename internal::VectorContainer<source_T, uint32_t>::difference_type,
                                                                                                            DenseHistogram<source_T>>
{
  using containerBase_type = internal::VectorContainer<source_T, uint32_t>;
  using HistogramConcept_type = internal::HistogramConcept<source_T,
                                                           typename internal::VectorContainer<source_T, uint32_t>::value_type,
                                                           typename internal::VectorContainer<source_T, uint32_t>::difference_type,
                                                           DenseHistogram<source_T>>;

  friend containerBase_type;
  friend HistogramConcept_type;

 public:
  using source_type = source_T;
  using value_type = typename containerBase_type::value_type;
  using container_type = typename containerBase_type::container_type;
  using size_type = typename containerBase_type::size_type;
  using difference_type = typename containerBase_type::difference_type;
  using reference = typename containerBase_type::reference;
  using const_reference = typename containerBase_type::const_reference;
  using pointer = typename containerBase_type::pointer;
  using const_pointer = typename containerBase_type::const_pointer;
  using const_iterator = typename containerBase_type::const_iterator;

  DenseHistogram() = default;

  template <typename freq_IT>
  DenseHistogram(freq_IT begin, freq_IT end, difference_type offset) : containerBase_type(), HistogramConcept_type{begin, end, offset} {};

  // operations
  using HistogramConcept_type::addSamples;

  using HistogramConcept_type::addFrequencies;

  template <typename source_IT>
  DenseHistogram& addSamples(source_IT begin, source_IT end, source_type min, source_type max)
  {
    if constexpr (std::is_pointer_v<source_IT>) {
      return addSamplesImpl(gsl::make_span(begin, end), min, max);
    } else {
      return addSamplesImpl(begin, end, min, max);
    }
  };

  template <typename source_IT>
  DenseHistogram& addSamples(gsl::span<const source_type> span, source_type min, source_type max)
  {
    return addSamplesImpl(span, min, max);
  };

  DenseHistogram& resize(source_type min, source_type max);

  inline DenseHistogram& resize(size_type newSize)
  {
    return resize(this->getOffset(), this->getOffset() + newSize);
  };

 protected:
  DenseHistogram& addSamplesImpl(gsl::span<const source_type> span);

  template <typename source_IT>
  DenseHistogram& addSamplesImpl(source_IT begin, source_IT end);

  template <typename source_IT>
  DenseHistogram& addSamplesImpl(source_IT begin, source_IT end, source_type min, source_type max);

  DenseHistogram& addSamplesImpl(gsl::span<const source_type> span, source_type min, source_type max);

  template <typename freq_IT>
  DenseHistogram& addFrequenciesImpl(freq_IT begin, freq_IT end, difference_type offset);

 private:
  inline static constexpr size_t MaxSize{utils::pow2(30)}; // 4GB per histogram size limit;

  bool isValidRange(difference_type min, difference_type max);
};

template <typename source_T>
inline bool DenseHistogram<source_T, std::enable_if_t<sizeof(source_T) == 4>>::isValidRange(difference_type min, difference_type max)
{
  bool ret = true;
  if constexpr (std::is_unsigned_v<source_T>) {
    if (min < 0) {
      LOGP(warning, "trying to add frequencies for a signed symbol to a DenseHistogram of an unsiged type.");
      ret = false;
    }
  }
  if (max - min > static_cast<difference_type>(this->MaxSize)) {
    LOGP(warning, "DenseHistogram exceeds {} elements threshold", this->MaxSize);
    ret = false;
  }
  return ret;
}

template <typename source_T>
inline auto DenseHistogram<source_T, std::enable_if_t<sizeof(source_T) == 4>>::addSamplesImpl(gsl::span<const source_type> samples) -> DenseHistogram&
{
  if (samples.size() > 0) {
    const auto [min, max] = internal::minmax(samples);
    addSamplesImpl(samples, min, max);
  } else {
    LOG(warning) << "Passed empty message to " << __func__; // RS this is ok for empty columns
  }
  return *this;
}

template <typename source_T>
template <typename source_IT>
inline auto DenseHistogram<source_T, std::enable_if_t<sizeof(source_T) == 4>>::addSamplesImpl(source_IT begin, source_IT end) -> DenseHistogram&
{
  if (begin != end) {
    const auto [minIter, maxIter] = std::minmax_element(begin, end);
    addSamplesImpl(begin, end, *minIter, *maxIter);
  } else {
    LOG(warning) << "Passed empty message to " << __func__; // RS this is ok for empty columns
  }
  return *this;
}

template <typename source_T>
inline auto DenseHistogram<source_T, std::enable_if_t<sizeof(source_T) == 4>>::addSamplesImpl(gsl::span<const source_type> samples, source_type min, source_type max) -> DenseHistogram&
{
  using namespace internal;
  using namespace utils;

  if (samples.empty()) {
    return *this;
  }
  this->resize(min, max);

  const auto begin = samples.data();
  const auto end = begin + samples.size();
  constexpr size_t ElemsPerQWord = sizeof(uint64_t) / sizeof(source_type);
  constexpr size_t nUnroll = 4 * ElemsPerQWord;
  auto iter = begin;

  if (getRangeBits(min, max) <= 17) {
    container_type histogram{this->mContainer.size(), this->mContainer.getOffset()};

    auto addQWord = [&, this](uint64_t in64) {
      uint64_t i = in64;
      ++this->mContainer[static_cast<source_type>(i)];
      i = in64 >> 32;
      ++histogram[static_cast<source_type>(i)];
    };

    if (end - nUnroll > begin) {
      for (; iter < end - nUnroll; iter += nUnroll) {
        addQWord(load64(iter));
        addQWord(load64(iter + ElemsPerQWord));
        addQWord(load64(iter + 2 * ElemsPerQWord));
        addQWord(load64(iter + 3 * ElemsPerQWord));
        this->mNSamples += nUnroll;
        __builtin_prefetch(iter + 512, 0);
      }
    }

    while (iter != end) {
      ++this->mNSamples;
      ++this->mContainer[*iter++];
    }
#if defined(RANS_OPENMP) && defined(RANS_SIMD)
#pragma omp simd
#endif
    for (size_t i = 0; i < this->size(); ++i) {
      this->mContainer.data()[i] += histogram.data()[i];
    }
  } else {
    auto addQWord = [&, this](uint64_t in64) {
      uint64_t i = in64;
      ++this->mContainer[static_cast<source_type>(i)];
      i = in64 >> 32;
      ++this->mContainer[static_cast<source_type>(i)];
    };

    if (end - nUnroll > begin) {
      for (; iter < end - nUnroll; iter += nUnroll) {
        addQWord(load64(iter));
        addQWord(load64(iter + ElemsPerQWord));
        addQWord(load64(iter + 2 * ElemsPerQWord));
        addQWord(load64(iter + 3 * ElemsPerQWord));
        this->mNSamples += nUnroll;
        __builtin_prefetch(iter + 512, 0);
      }
    }

    while (iter != end) {
      ++this->mNSamples;
      ++this->mContainer[*iter++];
    }
  }

  return *this;
};

template <typename source_T>
template <typename source_IT>
auto DenseHistogram<source_T, std::enable_if_t<sizeof(source_T) == 4>>::addSamplesImpl(source_IT begin, source_IT end, source_type min, source_type max) -> DenseHistogram&
{
  if (begin == end) {
    LOG(warning) << "Passed empty message to " << __func__; // RS this is ok for empty columns
  } else {

    if (!this->isValidRange(min, max)) {
      throw HistogramError(fmt::format("Incompatible Frequency table dimensions: Cannot add samples in range [{},{}] to {} int32 histogram.",
                                       min, max, []() {if constexpr (std::is_signed_v<source_T>) {return "signed";} else {return "unsigned";} }()));
    }
    this->resize(min, max);
    // add new symbols
    std::for_each(begin, end, [this](source_type symbol) {
      ++this->mContainer[symbol];
      ++this->mNSamples; });
  }

  return *this;
}

template <typename source_T>
template <typename freq_IT>
auto DenseHistogram<source_T, std::enable_if_t<sizeof(source_T) == 4>>::addFrequenciesImpl(freq_IT begin, freq_IT end, difference_type offset) -> DenseHistogram&
{
  using namespace internal;

  const auto thisHistogramView = makeHistogramView(this->mContainer);
  const auto addedHistogramView = trim(HistogramView{begin, end, offset});
  if (addedHistogramView.empty()) {
    LOG(warning) << "Passed empty DenseHistogram to " << __func__; // RS this is ok for empty columns
  } else {

    const difference_type newMin = std::min(thisHistogramView.getMin(), addedHistogramView.getMin());
    const difference_type newMax = std::max(thisHistogramView.getMax(), addedHistogramView.getMax());

    if (!this->isValidRange(newMin, newMax) ||
        newMin < static_cast<difference_type>(std::numeric_limits<source_T>::min()) ||
        newMax > static_cast<difference_type>(std::numeric_limits<source_T>::max())) {
      throw HistogramError(fmt::format("Incompatible Frequency table dimensions: Cannot add [{},{}] to {} int32 histogram.",
                                       addedHistogramView.getMin(),
                                       addedHistogramView.getMax(),
                                       []() {if constexpr (std::is_signed_v<source_T>) {return "signed";} else {return "unsigned";} }()));
    };

    if (thisHistogramView.empty()) {
      this->mContainer = container_type(addedHistogramView.size(), addedHistogramView.getOffset());
      std::transform(addedHistogramView.begin(), addedHistogramView.end(), this->mContainer.begin(), [this](count_t frequency) {
        return this->countSamples(frequency);
      });
    } else {
      const difference_type newSize = newMax - newMin + 1;
      typename container_type::container_type newHistogram(newSize, 0);
      const auto newHistogramView = makeHistogramView(newHistogram, newMin);
      auto histogramOverlap = getIntersection(newHistogramView, thisHistogramView);
      assert(!histogramOverlap.empty());
      assert(histogramOverlap.size() == thisHistogramView.size());
      std::copy(thisHistogramView.begin(), thisHistogramView.end(), histogramOverlap.begin());

      histogramOverlap = getIntersection(newHistogramView, addedHistogramView);
      assert(!histogramOverlap.empty());
      assert(histogramOverlap.size() == addedHistogramView.size());
      std::transform(addedHistogramView.begin(), addedHistogramView.end(),
                     histogramOverlap.begin(), histogramOverlap.begin(),
                     [this](const count_t& a, const count_t& b) { return safeadd(this->countSamples(a), b); });

      this->mContainer = container_type{std::move(newHistogram), static_cast<source_type>(newHistogramView.getOffset())};
    }
  }

  return *this;
}

template <typename source_T>
auto DenseHistogram<source_T, std::enable_if_t<sizeof(source_T) == 4>>::resize(source_type min, source_type max) -> DenseHistogram&
{
  using namespace utils;

  auto getMaxSymbol = [this]() {
    return static_cast<source_type>(this->getOffset() + std::max(0l, static_cast<int32_t>(this->size()) - 1l));
  };

  min = std::min(min, this->getOffset());
  max = std::max(max, getMaxSymbol());

  if (min > max) {
    throw HistogramError(fmt::format("{} failed: min {} > max {} ", __func__, min, max));
  }

  const size_type newSize = max - min + 1;
  const source_type oldOffset = this->getOffset();
  this->mNSamples = 0;

  if (this->mContainer.empty()) {
    this->mContainer = container_type{newSize, min};
    return *this;
  } else {
    container_type oldHistogram = std::move(this->mContainer);
    const auto oldHistogramView = makeHistogramView(oldHistogram, oldOffset);
    this->mContainer = container_type{newSize, min};
    return this->addFrequencies(oldHistogramView.begin(), oldHistogramView.end(), oldHistogramView.getMin());
  }
}

template <typename source_T>
class DenseHistogram<source_T, std::enable_if_t<sizeof(source_T) <= 2>> : public internal::VectorContainer<source_T, uint32_t>,
                                                                          internal::HistogramConcept<source_T,
                                                                                                     typename internal::VectorContainer<source_T, uint32_t>::value_type,
                                                                                                     typename internal::VectorContainer<source_T, uint32_t>::difference_type,
                                                                                                     DenseHistogram<source_T>>
{
  using containerBase_type = internal::VectorContainer<source_T, uint32_t>;
  using HistogramConcept_type = internal::HistogramConcept<source_T,
                                                           typename internal::VectorContainer<source_T, uint32_t>::value_type,
                                                           typename internal::VectorContainer<source_T, uint32_t>::difference_type,
                                                           DenseHistogram<source_T>>;

  friend containerBase_type;
  friend HistogramConcept_type;

 public:
  using source_type = source_T;
  using value_type = typename containerBase_type::value_type;
  using container_type = typename containerBase_type::container_type;
  using size_type = typename containerBase_type::size_type;
  using difference_type = typename containerBase_type::difference_type;
  using reference = typename containerBase_type::reference;
  using const_reference = typename containerBase_type::const_reference;
  using pointer = typename containerBase_type::pointer;
  using const_pointer = typename containerBase_type::const_pointer;
  using const_iterator = typename containerBase_type::const_iterator;

  DenseHistogram() : containerBase_type{MaxSize, std::numeric_limits<source_type>::min()} {};

  template <typename freq_IT>
  DenseHistogram(freq_IT begin, freq_IT end, difference_type offset) : containerBase_type{MaxSize, std::numeric_limits<source_type>::min()},
                                                                       HistogramConcept_type{begin, end, offset} {};

  using HistogramConcept_type::addSamples;

  template <typename source_IT>
  inline DenseHistogram& addSamples(source_IT begin, source_IT end, source_type min, source_type max)
  {
    return addSamplesImpl(begin, end);
  };

  template <typename source_IT>
  DenseHistogram& addSamples(gsl::span<const source_type> span, source_type min, source_type max)
  {
    return addSamplesImpl(span);
  };

  using HistogramConcept_type::addFrequencies;

 protected:
  template <typename source_IT>
  DenseHistogram& addSamplesImpl(source_IT begin, source_IT end);

  DenseHistogram& addSamplesImpl(gsl::span<const source_type> samples);

  template <typename freq_IT>
  DenseHistogram& addFrequenciesImpl(freq_IT begin, freq_IT end, difference_type offset);

 private:
  inline static constexpr size_t MaxSize = utils::pow2(utils::toBits<source_type>());
};

template <typename source_T>
template <typename source_IT>
auto DenseHistogram<source_T, std::enable_if_t<sizeof(source_T) <= 2>>::addSamplesImpl(source_IT begin, source_IT end) -> DenseHistogram&
{
  if constexpr (std::is_pointer_v<source_IT>) {
    return addSamplesImpl({begin, end});
  } else {
    std::for_each(begin, end, [this](const source_type& symbol) {
      ++this->mNSamples;
      ++this->mContainer[symbol]; });
  }
  return *this;
}

template <typename source_T>
auto DenseHistogram<source_T, std::enable_if_t<sizeof(source_T) <= 2>>::addSamplesImpl(gsl::span<const source_type> samples) -> DenseHistogram&
{
  using namespace internal;
  using namespace utils;

  if (samples.empty()) {
    return *this;
  }

  const auto begin = samples.data();
  const auto end = begin + samples.size();
  constexpr size_t ElemsPerQWord = sizeof(uint64_t) / sizeof(source_type);
  constexpr size_t nUnroll = 2 * ElemsPerQWord;
  auto iter = begin;

  if constexpr (sizeof(source_type) == 1) {

    std::array<ShiftableVector<source_type, value_type>, 3> histograms{
      {{this->mContainer.size(), this->mContainer.getOffset()},
       {this->mContainer.size(), this->mContainer.getOffset()},
       {this->mContainer.size(), this->mContainer.getOffset()}}};

    auto addQWord = [&, this](uint64_t in64) {
      uint64_t i = in64;
      ++histograms[0][static_cast<source_type>(i)];
      ++histograms[1][static_cast<source_type>(static_cast<uint16_t>(i) >> 8)];
      i >>= 16;
      ++histograms[2][static_cast<source_type>(i)];
      ++this->mContainer[static_cast<source_type>(static_cast<uint16_t>(i) >> 8)];
      i = in64 >>= 32;
      ++histograms[0][static_cast<source_type>(i)];
      ++histograms[1][static_cast<source_type>(static_cast<uint16_t>(i) >> 8)];
      i >>= 16;
      ++histograms[2][static_cast<source_type>(i)];
      ++this->mContainer[static_cast<source_type>(static_cast<uint16_t>(i) >> 8)];
    };

    if (end - nUnroll > begin) {
      for (; iter < end - nUnroll; iter += nUnroll) {
        addQWord(load64(iter));
        addQWord(load64(iter + ElemsPerQWord));
        this->mNSamples += nUnroll;
        __builtin_prefetch(iter + 512, 0);
      }
    }

    while (iter != end) {
      ++this->mNSamples;
      ++this->mContainer[*iter++];
    }

#if defined(RANS_OPENMP)
#pragma omp unroll full
#endif
    for (size_t j = 0; j < 3; ++j) {
#if defined(RANS_OPENMP) && defined(RANS_SIMD)
#pragma omp simd
#endif
      for (size_t i = 0; i < 256; ++i) {
        this->mContainer(i) += histograms[j](i);
      }
    }
  } else {
    container_type histogram{this->mContainer.size(), this->mContainer.getOffset()};

    auto addQWord = [&, this](uint64_t in64) {
      uint64_t i = in64;
      ++histogram[static_cast<source_type>(i)];
      ++this->mContainer[static_cast<source_type>(static_cast<uint32_t>(i) >> 16)];
      i = in64 >> 32;
      ++histogram[static_cast<source_type>(i)];
      ++this->mContainer[static_cast<source_type>(static_cast<uint32_t>(i) >> 16)];
    };

    if (end - nUnroll > begin) {
      for (; iter < end - nUnroll; iter += nUnroll) {
        addQWord(load64(iter));
        addQWord(load64(iter + ElemsPerQWord));
        this->mNSamples += nUnroll;
        __builtin_prefetch(iter + 512, 0);
      }
    }

    while (iter != end) {
      ++this->mNSamples;
      ++this->mContainer[*iter++];
    }
#if defined(RANS_OPENMP) && defined(RANS_SIMD)
#pragma omp simd
#endif
    for (size_t i = 0; i < this->size(); ++i) {
      this->mContainer.data()[i] += histogram.data()[i];
    }
  }

  return *this;
}

template <typename source_T>
template <typename freq_IT>
auto DenseHistogram<source_T, std::enable_if_t<sizeof(source_T) <= 2>>::addFrequenciesImpl(freq_IT begin, freq_IT end, difference_type offset) -> DenseHistogram&
{
  using namespace internal;

  // bounds check
  HistogramView addedHistogramView{begin, end, offset};
  addedHistogramView = trim(addedHistogramView);

  if constexpr (std::is_unsigned_v<source_T>) {
    LOG_IF(warning, addedHistogramView.getMin() < 0) << fmt::format("trying to add frequencies of a signed symbol type to a DenseHistogram of an unsiged type.");
  }
  if constexpr (std::is_signed_v<source_T>) {
    const std::ptrdiff_t sourceTypeMax = std::numeric_limits<source_T>::max();
    const bool isMinOutOfRange = addedHistogramView.getMin() > sourceTypeMax;
    const bool isMaxOutOfRange = addedHistogramView.getMax() > sourceTypeMax;

    if (isMaxOutOfRange || isMinOutOfRange) {
      LOGP(warning, "trying to add frequencies of an unsigned symbol type to a DenseHistogram of a signed type");
    }
  }

  const auto thisHistogramView = makeHistogramView(this->mContainer);
  const bool isInvalidFrequencyRange = utils::toBytes(utils::getRangeBits(addedHistogramView.getMin(), addedHistogramView.getMax())) > sizeof(source_T);

  if (isInvalidFrequencyRange) {
    throw HistogramError(fmt::format("Incompatible Frequency table dimensions: Cannot add [{},{}] to [{}, {}] ",
                                     addedHistogramView.getMin(),
                                     addedHistogramView.getMax(),
                                     thisHistogramView.getMin(),
                                     thisHistogramView.getMax()));
  }

  auto idx = static_cast<source_type>(addedHistogramView.getOffset()); // We tolerate uint8_t <-> int8_t and uint16_t <-> int16_t conversion.
                                                                       // The resulting overflows are handled correctly by the container via [] operator.
  for (freq_IT iter = addedHistogramView.begin(); iter != addedHistogramView.end(); ++iter) {
    auto frequency = *iter;
    this->mContainer[idx] = safeadd(this->mContainer[idx], this->countSamples(frequency));
    ++idx;
  }
  return *this;
}

template <typename source_T>
size_t countNUsedAlphabetSymbols(const DenseHistogram<source_T>& histogram)
{
  return std::count_if(histogram.begin(), histogram.end(), [](typename DenseHistogram<source_T>::const_reference v) { return v != typename DenseHistogram<source_T>::value_type{}; });
}

} // namespace o2::rans

#endif /* RANS_INTERNAL_CONTAINERS_DENSEHISTOGRAM_H_ */
