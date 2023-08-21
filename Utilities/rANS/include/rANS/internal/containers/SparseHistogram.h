// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FrequencyTableImpl.h
/// @author Michael Lettrich
/// @since  2019-05-08
/// @brief Histogram to depict frequencies of source symbols for rANS compression.

#ifndef INCLUDE_RANS_INTERNAL_CONTAINERS_SPARSEHISTOGRAM_H_
#define INCLUDE_RANS_INTERNAL_CONTAINERS_SPARSEHISTOGRAM_H_

#include "rANS/internal/common/utils.h"
#include "rANS/internal/containers/HistogramConcept.h"
#include "rANS/internal/containers/Container.h"

namespace o2::rans
{

template <typename source_T>
class SparseHistogram : public internal::SparseVectorContainer<source_T, uint32_t>,
                        public internal::HistogramConcept<source_T,
                                                          typename internal::SparseVectorContainer<source_T, uint32_t>::value_type,
                                                          typename internal::SparseVectorContainer<source_T, uint32_t>::difference_type,
                                                          SparseHistogram<source_T>>
{
  using containerBase_type = internal::SparseVectorContainer<source_T, uint32_t>;
  using HistogramConcept_type = internal::HistogramConcept<source_T,
                                                           typename internal::SparseVectorContainer<source_T, uint32_t>::value_type,
                                                           typename internal::SparseVectorContainer<source_T, uint32_t>::difference_type,
                                                           SparseHistogram<source_T>>;

  friend HistogramConcept_type;

 public:
  using source_type = typename containerBase_type::source_type;
  using value_type = typename containerBase_type::value_type;
  using container_type = typename containerBase_type::container_type;
  using size_type = typename containerBase_type::size_type;
  using difference_type = typename containerBase_type::difference_type;
  using reference = typename containerBase_type::reference;
  using const_reference = typename containerBase_type::const_reference;
  using pointer = typename containerBase_type::pointer;
  using const_pointer = typename containerBase_type::const_pointer;
  using const_iterator = typename containerBase_type::const_iterator;

  SparseHistogram() = default;

  template <typename freq_IT>
  SparseHistogram(freq_IT begin, freq_IT end, source_type offset) : containerBase_type(), HistogramConcept_type{begin, end, offset} {};

  // operations
  using HistogramConcept_type::addSamples;

  using HistogramConcept_type::addFrequencies;

 protected:
  template <typename source_IT>
  SparseHistogram& addSamplesImpl(source_IT begin, source_IT end);

  inline SparseHistogram& addSamplesImpl(gsl::span<const source_type> samples) { return addSamplesImpl(samples.data(), samples.data() + samples.size()); };

  template <typename freq_IT>
  SparseHistogram& addFrequenciesImpl(freq_IT begin, freq_IT end, source_type offset);
};

template <typename source_T>
template <typename source_IT>
auto SparseHistogram<source_T>::addSamplesImpl(source_IT begin, source_IT end) -> SparseHistogram&
{

  if constexpr (std::is_same_v<typename std::iterator_traits<source_IT>::iterator_category, std::random_access_iterator_tag>) {

    const auto size = std::distance(begin, end);
    constexpr size_t nUnroll = 2;

    size_t pos{};
    if (end - nUnroll > begin) {
      for (pos = 0; pos < size - nUnroll; pos += nUnroll) {
        ++this->mContainer[begin[pos + 0]];
        ++this->mContainer[begin[pos + 1]];
        this->mNSamples += nUnroll;
      }
    }

    for (auto iter = begin + pos; iter != end; ++iter) {
      ++this->mNSamples;
      ++this->mContainer[*iter];
    }
  } else {
    std::for_each(begin, end, [this](const source_type& symbol) {
      ++this->mNSamples;
      ++this->mContainer[symbol];
    });
  }
  return *this;
}

template <typename source_T>
template <typename freq_IT>
auto SparseHistogram<source_T>::addFrequenciesImpl(freq_IT begin, freq_IT end, source_type offset) -> SparseHistogram&
{
  source_type sourceSymbol = offset;
  for (auto iter = begin; iter != end; ++iter) {
    auto value = *iter;
    if (value > 0) {
      auto& currentValue = this->mContainer[sourceSymbol];
      currentValue = internal::safeadd(currentValue, this->countSamples(value));
    }
    ++sourceSymbol;
  }
  return *this;
}

template <typename source_T>
size_t countNUsedAlphabetSymbols(const SparseHistogram<source_T>& histogram)
{
  using iterator_value_type = typename SparseHistogram<source_T>::const_iterator::value_type;
  using value_type = typename SparseHistogram<source_T>::value_type;

  return std::count_if(histogram.begin(), histogram.end(), [](iterator_value_type v) { return v.second != value_type{}; });
}

} // namespace o2::rans

#endif /* INCLUDE_RANS_INTERNAL_CONTAINERS_SPARSEHISTOGRAM_H_ */
