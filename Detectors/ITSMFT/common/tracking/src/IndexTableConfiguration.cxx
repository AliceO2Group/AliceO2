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

#include "ITSMFTTracking/IndexTableConfiguration.h"

#include <array>
#include <cstdint>
#include <limits>

#include "CommonConstants/MathConstants.h"
#include "GPUCommonMath.h"

namespace o2::itsmft::tracking
{

using o2::itsmft::IndexTableCoordType;

IndexTableConfigError bindIndexTableConfiguration(o2::itsmft::IndexTableUtilsCore& staged,
                                                  const DetectorParameters& params,
                                                  int activeSurfaceCount,
                                                  SurfaceKind kind,
                                                  gsl::span<const SurfaceChartRange> chartRanges) noexcept
{
  if (kind != SurfaceKind::Cylinder && kind != SurfaceKind::Disk) {
    return IndexTableConfigError::InvalidSurfaceKind;
  }
  if (!(activeSurfaceCount > 0 && activeSurfaceCount <= o2::itsmft::IndexTableUtilsCore::MaxLayers)) {
    return IndexTableConfigError::InvalidActiveLayerCount;
  }
  if (params.RowBins <= 0) {
    return IndexTableConfigError::NonPositiveRowBins;
  }
  if (params.ColBins <= 0) {
    return IndexTableConfigError::NonPositiveColBins;
  }

  const std::uint64_t binCount = static_cast<std::uint64_t>(params.RowBins) * static_cast<std::uint64_t>(params.ColBins);
  if (binCount > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    return IndexTableConfigError::RowColBinCountExceedsIndexRange;
  }

  if (chartRanges.size() < static_cast<std::size_t>(activeSurfaceCount)) {
    return IndexTableConfigError::InsufficientChartRanges;
  }
  std::array<float, o2::itsmft::IndexTableUtilsCore::MaxLayers> colMin{};
  std::array<float, o2::itsmft::IndexTableUtilsCore::MaxLayers> colMax{};
  for (int iLayer = 0; iLayer < activeSurfaceCount; ++iLayer) {
    if (!o2::gpu::GPUCommonMath::Finite(chartRanges[iLayer].min) ||
        !o2::gpu::GPUCommonMath::Finite(chartRanges[iLayer].max)) {
      return IndexTableConfigError::NonFiniteChartRange;
    }
    if (!(chartRanges[iLayer].max > chartRanges[iLayer].min)) {
      return IndexTableConfigError::InvalidChartRange;
    }
    colMin[iLayer] = chartRanges[iLayer].min;
    colMax[iLayer] = chartRanges[iLayer].max;
  }

  staged.setIndexTableParams(kind == SurfaceKind::Disk ? IndexTableCoordType::PhiR : IndexTableCoordType::PhiZ,
                             params.RowBins, params.ColBins, 0.f, o2::constants::math::TwoPI,
                             gsl::span<const float>{colMin.data(), static_cast<std::size_t>(activeSurfaceCount)},
                             gsl::span<const float>{colMax.data(), static_cast<std::size_t>(activeSurfaceCount)});
  return IndexTableConfigError::None;
}

} // namespace o2::itsmft::tracking
