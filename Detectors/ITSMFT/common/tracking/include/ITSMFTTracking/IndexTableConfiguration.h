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

#ifndef ALICEO2_ITSMFT_TRACKING_INDEXTABLECONFIGURATION_H_
#define ALICEO2_ITSMFT_TRACKING_INDEXTABLECONFIGURATION_H_

#include <cstdint>

// Host-only: DetectorParameters owns std::vector members and is not
// device-compatible. Keep this boundary separate so existing host-binding
// consumers do not inherit IndexTableUtils.h's extra dependencies.
#ifndef GPUCA_GPUCODE

#include <cmath>
#include <cstddef>
#include <limits>

#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/IndexTableUtils.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"

namespace o2::itsmft::tracking
{

enum class IndexTableConfigError : uint8_t {
  None,
  NonPositiveRowBins,
  NonPositiveColBins,
  RowColBinCountExceedsIndexRange, // Product exceeds int, the bin-index type.
  InvalidActiveLayerCount,         // Invalid active surface count.
  InsufficientChartRanges,         // Fewer descriptor chart ranges than active surfaces.
  NonFiniteChartRange,             // Chart bound is NaN or +/-Inf.
  InvalidChartRange,               // Chart maximum does not exceed its minimum.
  InvalidSurfaceKind,              // Neither Cylinder nor Disk.
};

/// Validates and binds detector inputs into `staged` for one coordinate kind.
/// Resolve `kind` from the validated DetectorLayout, never from NLayers or DetId.
/// On error, `staged` is unchanged. Call once per present kind during detector
/// initialization, outside iteration and candidate loops.
IndexTableConfigError bindIndexTableConfiguration(o2::itsmft::IndexTableUtilsCore& staged,
                                                  const DetectorParameters& params,
                                                  int activeSurfaceCount,
                                                  SurfaceKind kind,
                                                  gsl::span<const SurfaceChartRange> chartRanges) noexcept;

/// True iff all fields stored by setIndexTableParams match between `a` and
/// `b`. Used to verify that a non-FirstPass iteration matches the
/// TimeFrame-owned configuration before reusing or resorting its LUT.
inline bool indexTableConfigurationsMatch(const o2::itsmft::IndexTableUtilsCore& a,
                                          const o2::itsmft::IndexTableUtilsCore& b,
                                          int activeSurfaceCount) noexcept
{
  if (a.getCoordType() != b.getCoordType() ||
      a.getNrowBins() != b.getNrowBins() ||
      a.getNcolBins() != b.getNcolBins() ||
      a.getRowOrigin() != b.getRowOrigin() ||
      a.getRowCoordinateSpan() != b.getRowCoordinateSpan()) {
    return false;
  }
  if (activeSurfaceCount <= 0 || activeSurfaceCount > o2::itsmft::IndexTableUtilsCore::MaxLayers) {
    return false;
  }
  for (int iLayer = 0; iLayer < activeSurfaceCount; ++iLayer) {
    if (a.getLayerColMin(iLayer) != b.getLayerColMin(iLayer) ||
        a.getLayerColMax(iLayer) != b.getLayerColMax(iLayer)) {
      return false;
    }
  }
  return true;
}

/// Checked size_t multiplication for index-table allocation sizes. Returns
/// false, leaving `result` unset, if `a * b` overflows size_t.
inline bool checkedIndexTableSizeProduct(std::size_t a, std::size_t b, std::size_t& result) noexcept
{
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    return false;
  }
  result = a * b;
  return true;
}

} // namespace o2::itsmft::tracking

#endif // GPUCA_GPUCODE

#endif /* ALICEO2_ITSMFT_TRACKING_INDEXTABLECONFIGURATION_H_ */
