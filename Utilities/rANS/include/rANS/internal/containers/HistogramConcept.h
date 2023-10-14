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

/// @file   HistogramConcept.h
/// @author Michael Lettrich
/// @brief  Operations that will be performed on a histogram

#ifndef RANS_INTERNAL_CONTAINERS_HISTOGRAMCONCEPT_H_
#define RANS_INTERNAL_CONTAINERS_HISTOGRAMCONCEPT_H_

#include <gsl/span>

#include "rANS/internal/common/utils.h"

namespace o2::rans::internal
{

template <typename source_T, typename value_T, typename difference_T, class derived_T>
class HistogramConcept
{
  friend derived_T;

 public:
  using source_type = source_T;
  using value_type = value_T;
  using difference_type = difference_T;

  // operations
  template <typename source_IT>
  inline derived_T& addSamples(source_IT begin, source_IT end)
  {
    static_assert(utils::isCompatibleIter_v<source_type, source_IT>);

    if (begin == end) {
      return static_cast<derived_T&>(*this);
    } else {
      return static_cast<derived_T*>(this)->addSamplesImpl(begin, end);
    }
  };

  inline derived_T& addSamples(gsl::span<const source_type> samples)
  {
    return addSamples(samples.data(), samples.data() + samples.size());
  };

  template <typename freq_IT>
  inline derived_T& addFrequencies(freq_IT begin, freq_IT end, difference_type offset)
  {
    static_assert(utils::isCompatibleIter_v<value_type, freq_IT>);

    if (begin == end) {
      return static_cast<derived_T&>(*this);
    } else {
      return static_cast<derived_T*>(this)->addFrequenciesImpl(begin, end, offset);
    }
  };

  inline derived_T& addFrequencies(gsl::span<const value_type> frequencies, difference_type offset)
  {
    return addFrequencies(frequencies.data(), frequencies.data() + frequencies.size(), offset);
  };

  derived_T& operator+(derived_T& other)
  {
    return addFrequencies(other.cbegin(), other.cbegin(), other.getOffset());
  };

 protected:
  HistogramConcept() = default;

  template <typename freq_IT>
  HistogramConcept(freq_IT begin, freq_IT end, difference_type offset)
  {
    static_assert(utils::isIntegralIter_v<freq_IT>);
    addFrequencies(begin, end, offset);
  };
};

} // namespace o2::rans::internal

#endif /* RANS_INTERNAL_CONTAINERS_HISTOGRAMCONCEPT_H_ */
