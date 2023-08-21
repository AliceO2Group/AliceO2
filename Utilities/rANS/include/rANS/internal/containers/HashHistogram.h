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

#ifndef INCLUDE_RANS_INTERNAL_CONTAINERS_HASHHISTOGRAM_H_
#define INCLUDE_RANS_INTERNAL_CONTAINERS_HASHHISTOGRAM_H_

#include "rANS/internal/common/utils.h"
#include "rANS/internal/containers/HistogramConcept.h"
#include "rANS/internal/containers/Container.h"

namespace o2::rans
{

template <typename source_T>
class HashHistogram : public internal::HashContainer<source_T, uint32_t>,
                      public internal::HistogramConcept<source_T,
                                                        typename internal::HashContainer<source_T, uint32_t>::value_type,
                                                        typename internal::HashContainer<source_T, uint32_t>::difference_type,
                                                        HashHistogram<source_T>>
{
  using containerBase_type = internal::HashContainer<source_T, uint32_t>;
  using HistogramConcept_type = internal::HistogramConcept<source_T,
                                                           typename internal::HashContainer<source_T, uint32_t>::value_type,
                                                           typename internal::HashContainer<source_T, uint32_t>::difference_type,
                                                           HashHistogram<source_T>>;

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

  HashHistogram() = default;

  template <typename freq_IT>
  HashHistogram(freq_IT begin, freq_IT end, source_type offset) : containerBase_type(), HistogramConcept_type{begin, end, offset} {};

  // operations
  using HistogramConcept_type::addSamples;

  using HistogramConcept_type::addFrequencies;

 protected:
  template <typename source_IT>
  HashHistogram& addSamplesImpl(source_IT begin, source_IT end);

  inline HashHistogram& addSamplesImpl(gsl::span<const source_type> samples) { return addSamplesImpl(samples.data(), samples.data() + samples.size()); };

  template <typename freq_IT>
  HashHistogram& addFrequenciesImpl(freq_IT begin, freq_IT end, source_type offset);
};

template <typename source_T>
template <typename source_IT>
auto HashHistogram<source_T>::addSamplesImpl(source_IT begin, source_IT end) -> HashHistogram&
{
  std::for_each(begin, end, [this](const source_type& symbol) {
    ++this->mNSamples;
    ++this->mContainer[symbol];
  });
  return *this;
}

template <typename source_T>
template <typename freq_IT>
auto HashHistogram<source_T>::addFrequenciesImpl(freq_IT begin, freq_IT end, source_type offset) -> HashHistogram&
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
size_t countNUsedAlphabetSymbols(const HashHistogram<source_T>& histogram)
{
  using const_reference = const typename std::iterator_traits<typename HashHistogram<source_T>::const_iterator>::value_type&;
  using value_type = typename HashHistogram<source_T>::value_type;

  return std::count_if(histogram.begin(), histogram.end(), [](const_reference v) { return v.second != value_type{}; });
}

} // namespace o2::rans

#endif /* INCLUDE_RANS_INTERNAL_CONTAINERS_HASHHISTOGRAM_H_ */
