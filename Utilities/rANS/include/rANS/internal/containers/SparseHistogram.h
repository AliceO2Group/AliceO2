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

/// @file   SparseHistogram.h
/// @author Michael Lettrich
/// @brief Histogram to depict frequencies of source symbols for rANS compression, based on an ordered set

#ifndef INCLUDE_RANS_INTERNAL_CONTAINERS_SPARSEHISTOGRAM_H_
#define INCLUDE_RANS_INTERNAL_CONTAINERS_SPARSEHISTOGRAM_H_

#include "rANS/internal/common/utils.h"
#include "rANS/internal/containers/HistogramConcept.h"
#include "rANS/internal/containers/Container.h"

namespace o2::rans
{

template <typename source_T>
class SparseHistogram : public internal::SetContainer<source_T, uint32_t>,
                        public internal::HistogramConcept<source_T,
                                                          typename internal::SetContainer<source_T, uint32_t>::value_type,
                                                          typename internal::SetContainer<source_T, uint32_t>::difference_type,
                                                          SparseHistogram<source_T>>
{
  using containerBase_type = internal::SetContainer<source_T, uint32_t>;
  using HistogramConcept_type = internal::HistogramConcept<source_T,
                                                           typename internal::SetContainer<source_T, uint32_t>::value_type,
                                                           typename internal::SetContainer<source_T, uint32_t>::difference_type,
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

  absl::flat_hash_map<source_type, value_type> map;

  // first build a hash map of existing samples
  std::for_each(this->mContainer.begin(), this->mContainer.end(), [this, &map](const auto& keyValuePair) {
    map.emplace(keyValuePair.first, keyValuePair.second);
  });

  // then add new samples to the same map
  std::for_each(begin, end, [this, &map](const source_type& symbol) {
    ++this->mNSamples;
    ++map[symbol];
  });

  //
  typename container_type::container_type mergedSymbols;
  mergedSymbols.reserve(map.size());
  std::for_each(map.begin(), map.end(), [&](const auto& keyValuePair) { mergedSymbols.emplace_back(keyValuePair.first, keyValuePair.second); });

  // and build a OrderedSet from it
  this->mContainer = container_type(std::move(mergedSymbols), 0, internal::OrderedSetState::unordered);

  return *this;
}

template <typename source_T>
template <typename freq_IT>
auto SparseHistogram<source_T>::addFrequenciesImpl(freq_IT begin, freq_IT end, source_type offset) -> SparseHistogram&
{
  if (begin != end) {
    // first build a map of the current list items
    absl::flat_hash_map<source_type, value_type> map;

    auto container = std::move(this->mContainer).release();
    for (const auto& [key, value] : container) {
      map.emplace(key, value);
    }

    // then add all new values to the map
    source_type sourceSymbol = offset;
    for (auto iter = begin; iter != end; ++iter) {
      auto value = *iter;
      if (value > 0) {
        auto& currentValue = map[sourceSymbol];
        currentValue = internal::safeadd(currentValue, this->countSamples(value));
      }
      ++sourceSymbol;
    }

    // then extract key/value pairs
    container.clear();
    container.reserve(map.size());
    std::for_each(map.begin(), map.end(), [&](const auto& pair) { container.emplace_back(pair.first, pair.second); });

    // and build a OrderedSet from it
    this->mContainer = container_type{container, 0, internal::OrderedSetState::unordered};
  }
  return *this;
};

template <typename source_T>
size_t countNUsedAlphabetSymbols(const SparseHistogram<source_T>& histogram)
{
  using value_type = typename SparseHistogram<source_T>::value_type;

  return std::count_if(histogram.begin(), histogram.end(), [](const auto& v) { return v.second != value_type{}; });
}

} // namespace o2::rans

#endif /* INCLUDE_RANS_INTERNAL_CONTAINERS_SPARSEHISTOGRAM_H_ */
