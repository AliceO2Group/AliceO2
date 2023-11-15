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

/// @file   properties.h
/// @author Michael Lettrich
/// @brief  Essential properties of the dataset used for building dictionaries and coders

#ifndef RANS_INTERNAL_METRICS_PROPERTIES_H_
#define RANS_INTERNAL_METRICS_PROPERTIES_H_

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <array>
#include <optional>

#include "rANS/internal/metrics/DictSizeEstimate.h"

namespace o2::rans
{

template <typename source_T>
struct CoderProperties {
  using source_type = source_T;

  internal::DictSizeEstimate dictSizeEstimate{};
  std::optional<size_t> renormingPrecisionBits{};
  std::optional<size_t> nIncompressibleSymbols{};
  std::optional<size_t> nIncompressibleSamples{};
  std::optional<source_type> min{};
  std::optional<source_type> max{};
};

template <typename source_T>
struct DatasetProperties {
  using source_type = source_T;

  source_type min{};
  source_type max{};
  size_t numSamples{};
  uint32_t alphabetRangeBits{};
  uint32_t nUsedAlphabetSymbols{};
  float_t entropy{};
  std::array<uint32_t, 32> symbolLengthDistribution{{}};
  std::array<uint32_t, 32> weightedSymbolLengthDistribution{{}};
};

} // namespace o2::rans

#endif /* RANS_INTERNAL_METRICS_PROPERTIES_H_ */
