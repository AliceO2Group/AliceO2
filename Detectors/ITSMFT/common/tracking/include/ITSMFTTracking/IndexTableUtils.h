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
///
/// \file IndexTableUtils.h
/// \brief Shared index-table utilities for periodic-phi surface charts
///

#ifndef ALICEO2_ITSMFT_TRACKING_INDEXTABLEUTILS_H_
#define ALICEO2_ITSMFT_TRACKING_INDEXTABLEUTILS_H_

#include <algorithm>
#include <array>
#include <cstddef>

#include <gsl/span>

#include "CommonConstants/MathConstants.h"
#include "GPUCommonMath.h"
#include "GPUCommonDef.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/IdTypes.h"

namespace o2::itsmft
{

enum class IndexTableCoordType : uint8_t { PhiZ,
                                           PhiR };

namespace index_table_utils
{
GPUhdi() float getNormalizedPhi(float phi)
{
  phi -= o2::constants::math::TwoPI * o2::gpu::GPUCommonMath::Floor(phi * (1.f / o2::constants::math::TwoPI));
  return phi;
}
} // namespace index_table_utils

/// Row/column LUT helper. Charts have periodic phi rows and a
/// descriptor-bounded linear column.
/// MaxLayoutSurfaces storage keeps GPUhdi() access device-portable; callers
/// must not query unpopulated runtime-plan positions.
class IndexTableUtilsCore
{
 public:
  static constexpr int MaxLayers = static_cast<int>(o2::itsmft::tracking::MaxLayoutSurfaces);

  /// Configure LUT geometry with a row interval and per-surface column intervals.
  /// `layerColHalfExtent` may be shorter than MaxLayers (the common case --
  /// real detectors have far fewer than 32 layers); anything beyond its size
  /// is left at its previous value, exactly as it would be untouched by a
  /// caller that never re-populates it.
  void setIndexTableParams(IndexTableCoordType coordType, int nRowBins, int nColBins,
                           float rowMin, float rowMax,
                           gsl::span<const float> layerColMin,
                           gsl::span<const float> layerColMax)
  {
    mCoordType = coordType;
    mRowOrigin = 0.f;
    mRowCoordinateSpan = rowMax - rowMin;
    mInverseRowBinSize = (mRowCoordinateSpan > 0.f) ? static_cast<float>(nRowBins) / mRowCoordinateSpan : 0.f;
    mNcolBins = nColBins;
    mNrowBins = nRowBins;
    const int nLayers = std::min({static_cast<int>(layerColMin.size()), static_cast<int>(layerColMax.size()), MaxLayers});
    for (int iLayer{0}; iLayer < nLayers; ++iLayer) {
      mLayerColMin[iLayer] = layerColMin[iLayer];
      mLayerColMax[iLayer] = layerColMax[iLayer];
      mInverseColBinSize[iLayer] = static_cast<float>(nColBins) / (layerColMax[iLayer] - layerColMin[iLayer]);
    }
  }

  void setIndexTableParams(IndexTableCoordType coordType, int nRowBins, int nColBins,
                           float rowMin, float rowMax,
                           gsl::span<const float> layerColHalfExtent)
  {
    std::array<float, MaxLayers> minima{};
    std::array<float, MaxLayers> maxima{};
    const int count = std::min(static_cast<int>(layerColHalfExtent.size()), MaxLayers);
    for (int iLayer = 0; iLayer < count; ++iLayer) {
      minima[iLayer] = -layerColHalfExtent[iLayer];
      maxima[iLayer] = layerColHalfExtent[iLayer];
    }
    setIndexTableParams(coordType, nRowBins, nColBins, rowMin, rowMax,
                        gsl::span<const float>{minima.data(), static_cast<size_t>(count)},
                        gsl::span<const float>{maxima.data(), static_cast<size_t>(count)});
  }

  /// Fill LUT geometry from any struct exposing RowBins, ColBins and LayerZ (ITS phi-z).
  template <class T>
  void setTrackingParameters(const T& params)
  {
    const auto extents = layerColHalfExtentFrom(params);
    setIndexTableParams(IndexTableCoordType::PhiZ, params.RowBins, params.ColBins,
                        0.f, o2::constants::math::TwoPI, gsl::span<const float>{extents.data(), static_cast<std::size_t>(extents.count)});
  }

  GPUhdi() float getInverseColCoordinate(const int layerIndex) const
  {
    return mInverseColBinSize[layerIndex];
  }

  GPUhdi() int getColBinIndex(const int layerIndex, const float colCoordinate) const
  {
    return (colCoordinate - mLayerColMin[layerIndex]) * mInverseColBinSize[layerIndex];
  }

  GPUhdi() int getRowBinIndex(const float rowCoordinate) const
  {
    return rowCoordinate * mInverseRowBinSize;
  }

  GPUhdi() int getBinIndex(const int colIndex, const int rowIndex) const
  {
    return o2::gpu::GPUCommonMath::Min(rowIndex * mNcolBins + colIndex, (mNcolBins * mNrowBins) - 1);
  }

  GPUhdi() int countRowSelectedBins(const int* indexTable, const int rowBinIndex,
                                    const int minColBinIndex, const int maxColBinIndex) const
  {
    const int firstBinIndex{getBinIndex(minColBinIndex, rowBinIndex)};
    const int maxBinIndex{firstBinIndex + maxColBinIndex - minColBinIndex + 1};

    return indexTable[maxBinIndex] - indexTable[firstBinIndex];
  }

  void print() const;

  GPUhdi() int getNcolBins() const { return mNcolBins; }
  GPUhdi() int getNrowBins() const { return mNrowBins; }
  GPUhdi() float getLayerColHalfExtent(int i) const { return 0.5f * (mLayerColMax[i] - mLayerColMin[i]); }
  GPUhdi() float getLayerColMin(int i) const { return mLayerColMin[i]; }
  GPUhdi() float getLayerColMax(int i) const { return mLayerColMax[i]; }
  GPUhdi() void setNcolBins(const int colBins) { mNcolBins = colBins; }
  GPUhdi() void setNrowBins(const int rowBins) { mNrowBins = rowBins; }
  GPUhdi() IndexTableCoordType getCoordType() const { return mCoordType; }
  /// Row origin/span, needed alongside the other getters to detect a
  /// configuration mismatch between a freshly bound IndexTableUtils and one
  /// already owned by a TimeFrame (LUT-reuse invariant); not test-only.
  GPUhdi() float getRowOrigin() const { return mRowOrigin; }
  GPUhdi() float getRowCoordinateSpan() const { return mRowCoordinateSpan; }

 private:
  /// Fixed-capacity result of layerColHalfExtentFrom(); count is the number of
  /// available entries, never above MaxLayers.
  struct LayerExtents {
    std::array<float, MaxLayers> values{};
    int count{0};
    const float* data() const noexcept { return values.data(); }
  };

  template <class T>
  static LayerExtents layerColHalfExtentFrom(const T& params)
  {
    LayerExtents extents;
    if constexpr (requires { params.LayerColHalfExtent; }) {
      const auto& colExtents = params.LayerColHalfExtent.empty() ? params.LayerZ : params.LayerColHalfExtent;
      extents.count = std::min(static_cast<int>(colExtents.size()), MaxLayers);
      for (int iLayer{0}; iLayer < extents.count; ++iLayer) {
        extents.values[iLayer] = colExtents[iLayer];
      }
    } else {
      extents.count = std::min(static_cast<int>(params.LayerZ.size()), MaxLayers);
      for (int iLayer{0}; iLayer < extents.count; ++iLayer) {
        extents.values[iLayer] = params.LayerZ[iLayer];
      }
    }
    return extents;
  }

  int mNcolBins = 0;
  int mNrowBins = 0;
  float mInverseRowBinSize = 0.f;
  float mRowOrigin = 0.f;
  float mRowCoordinateSpan = o2::constants::math::TwoPI;
  IndexTableCoordType mCoordType{IndexTableCoordType::PhiZ};
  std::array<float, MaxLayers> mLayerColMin{};
  std::array<float, MaxLayers> mLayerColMax{};
  std::array<float, MaxLayers> mInverseColBinSize{};
};

inline void IndexTableUtilsCore::print() const
{
  printf("NcolBins: %d, NrowBins: %d, InverseRowBinSize: %f\n", mNcolBins, mNrowBins, mInverseRowBinSize);
  for (int iLayer{0}; iLayer < MaxLayers; ++iLayer) {
    printf("Layer %d: ColRange: [%f, %f], InverseColBinSize: %f\n", iLayer, mLayerColMin[iLayer], mLayerColMax[iLayer], mInverseColBinSize[iLayer]);
  }
}

/// Coordinate-neutral periodic-phi lookup. The operation is not templated on
/// nLayers -- see IndexTableUtilsCore's own doc; callers supply the runtime
/// plan slot and the surface descriptor determines the column coordinate.
GPUhdi() int4 getBinsPhiColumn(float phi, const int layerIndex,
                               float col, float maxDeltaCol, float maxDeltaRow,
                               const IndexTableUtilsCore& utils)
{
  const float colRangeMin = col - maxDeltaCol;
  const float rowRangeMin = (maxDeltaRow > o2::constants::math::PI) ? 0.f : phi - maxDeltaRow;
  const float colRangeMax = col + maxDeltaCol;
  const float rowRangeMax = (maxDeltaRow > o2::constants::math::PI) ? o2::constants::math::TwoPI : phi + maxDeltaRow;

  if (colRangeMax < utils.getLayerColMin(layerIndex) ||
      colRangeMin > utils.getLayerColMax(layerIndex) || colRangeMin > colRangeMax) {
    return int4{-1, -1, -1, -1};
  }

  return int4{o2::gpu::GPUCommonMath::Max(0, utils.getColBinIndex(layerIndex, colRangeMin)),
              utils.getRowBinIndex(index_table_utils::getNormalizedPhi(rowRangeMin)),
              o2::gpu::GPUCommonMath::Min(utils.getNcolBins() - 1, utils.getColBinIndex(layerIndex, colRangeMax)),
              utils.getRowBinIndex(index_table_utils::getNormalizedPhi(rowRangeMax))};
}

} // namespace o2::itsmft

#endif /* ALICEO2_ITSMFT_TRACKING_INDEXTABLEUTILS_H_ */
