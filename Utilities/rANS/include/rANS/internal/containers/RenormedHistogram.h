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

#include "rANS/internal/containers/CountingContainer.h"
#include "rANS/internal/containers/HistogramView.h"

namespace o2::rans
{

template <typename source_T>
class RenormedHistogram : public internal::CountingContainer<source_T>
{
  using base_type = internal::CountingContainer<source_T>;

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
  using iterator = typename base_type::iterator;
  using const_reverse_iterator = typename base_type::const_reverse_iterator;
  using reverse_iterator = typename base_type::reverse_iterator;

  RenormedHistogram() : base_type(){};

  inline RenormedHistogram(container_type frequencies, size_t renormingBits, value_type nIncompressible) : mNIncompressible(nIncompressible)
  {
    this->mContainer = std::move(frequencies);
    this->mNSamples = utils::pow2(renormingBits);

#if !defined(NDEBUG)
    size_t nSamples = std::accumulate(this->begin(), this->end(), 0);
    nSamples += this->mNIncompressible;
    assert(internal::isPow2(nSamples));
    assert(nSamples == this->mNSamples);
#endif
  };

  [[nodiscard]] inline size_t getRenormingBits() const noexcept { return utils::log2UInt(this->mNSamples); };

  [[nodiscard]] inline bool isRenormedTo(size_t nBits) const noexcept { return nBits == this->getRenormingBits(); };

  [[nodiscard]] inline value_type getIncompressibleSymbolFrequency() const noexcept { return mNIncompressible; };

  [[nodiscard]] inline bool hasIncompressibleSymbol() const noexcept { return mNIncompressible != 0; };

  friend void swap(RenormedHistogram& a, RenormedHistogram& b) noexcept
  {
    using std::swap;
    swap(static_cast<typename RenormedHistogram::base_type&>(a),
         static_cast<typename RenormedHistogram::base_type&>(b));
  };

 private:
  value_type mNIncompressible{};
};

template <typename source_T>
std::pair<source_T, source_T> getMinMax(const RenormedHistogram<source_T>& histogram)
{
  auto view = trim(makeHistogramView(histogram));
  return {view.getMin(), view.getMax()};
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_CONTAINERS_RENORMEDHISTOGRAM_H_ */
