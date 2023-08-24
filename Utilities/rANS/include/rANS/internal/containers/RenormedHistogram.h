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

/// @file   RenormedHistogram.h
/// @author Michael Lettrich
/// @brief  Histogram renormed to sum of frequencies being 2^P for use in fast rans coding.

#ifndef RANS_INTERNAL_CONTAINERS_RENORMEDHISTOGRAM_H_
#define RANS_INTERNAL_CONTAINERS_RENORMEDHISTOGRAM_H_

#include <numeric>
#include <fairlogger/Logger.h>

#include "rANS/internal/containers/Container.h"
#include "rANS/internal/containers/HistogramView.h"

namespace o2::rans
{

template <class container_T>
class RenormedHistogramConcept : public container_T
{
  using base_type = container_T;

 public:
  using source_type = typename base_type::source_type;
  using value_type = typename base_type::value_type;
  using container_type = typename base_type::container_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;
  using reference = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using pointer = typename base_type::pointer;
  using const_pointer = typename base_type::const_pointer;
  using const_iterator = typename base_type::const_iterator;

  RenormedHistogramConcept() : base_type(){};

  inline RenormedHistogramConcept(container_type frequencies, size_t renormingBits, value_type nIncompressible) : mNIncompressible(nIncompressible)
  {
    this->mContainer = std::move(frequencies);
    this->mNSamples = utils::pow2(renormingBits);

#if !defined(NDEBUG)
    size_t nSamples = std::accumulate(this->begin(), this->end(), 0, [](const auto& a, const auto& b) {
      if constexpr (std::is_integral_v<std::remove_reference_t<decltype(b)>>) {
        return a + b;
      } else {
        return a + b.second;
      }
    });
    nSamples += this->mNIncompressible;
    assert(internal::isPow2(nSamples));
    assert(nSamples == this->mNSamples);
#endif
  };

  [[nodiscard]] inline size_t getRenormingBits() const noexcept { return utils::log2UInt(this->mNSamples); };

  [[nodiscard]] inline bool isRenormedTo(size_t nBits) const noexcept { return nBits == this->getRenormingBits(); };

  [[nodiscard]] inline value_type getIncompressibleSymbolFrequency() const noexcept { return mNIncompressible; };

  [[nodiscard]] inline bool hasIncompressibleSymbol() const noexcept { return mNIncompressible != 0; };

 private:
  value_type mNIncompressible{};
};

template <typename source_T>
using RenormedDenseHistogram = RenormedHistogramConcept<internal::VectorContainer<source_T, uint32_t>>;

template <typename source_T>
using RenormedAdaptiveHistogram = RenormedHistogramConcept<internal::SparseVectorContainer<source_T, uint32_t>>;

template <typename source_T>
using RenormedSparseHistogram = RenormedHistogramConcept<internal::SetContainer<source_T, uint32_t>>;

template <typename container_T>
size_t countNUsedAlphabetSymbols(const RenormedHistogramConcept<container_T>& histogram)
{
  return std::count_if(histogram.begin(), histogram.end(),
                       [](typename RenormedHistogramConcept<container_T>::const_reference v) {
                         return v != typename RenormedHistogramConcept<container_T>::value_type{};
                       });
}

} // namespace o2::rans

#endif /* RANS_INTERNAL_CONTAINERS_RENORMEDHISTOGRAM_H_ */
