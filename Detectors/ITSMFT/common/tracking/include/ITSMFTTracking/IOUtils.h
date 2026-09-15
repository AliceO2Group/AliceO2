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
/// \file IOUtils.h
/// \brief Shared cluster I/O utilities for ITS and MFT (based on ITStracking/IOUtils.h)
///

#ifndef ALICEO2_ITSMFT_TRACKING_IOUTILS_H_
#define ALICEO2_ITSMFT_TRACKING_IOUTILS_H_

#include <array>
#include <cstdint>
#include <limits>

#ifndef GPUCA_GPUCODE
#include <stdexcept>
#include <string>
#endif

#include <gsl/gsl>

#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSMFTTracking/ClusterDecoding.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/ROFViews.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/SurfaceMeasurement.h"
#include "ITSMFTTracking/SurfaceTiming.h"
#include "MathUtils/Cartesian.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2::itsmft::ioutils
{

namespace detail
{
constexpr bool isSensorInGeometry(int sensor, int geometrySize) noexcept
{
  return sensor >= 0 && sensor < geometrySize;
}

constexpr bool isLayerInDetector(int layer, int detectorLayers) noexcept
{
  return layer >= 0 && layer < detectorLayers;
}

/// Return whether cluster-decoding systematic errors are configured for `DetId`.
/// ITS is a no-op; MFT reads its live tracker configuration.
template <o2::detectors::DetID::ID DetId>
bool shouldApplySysErrors()
{
  if constexpr (DetId == o2::detectors::DetID::ITS) {
    return false;
  } else {
    const auto& conf = o2::itsmft::tracking::TrackerParamRef<DetId>::get();
    for (int il = 0; il < o2::itsmft::tracking::TrackerParamRef<DetId>::nLayers(); il++) {
      if (conf.sysErr2Row[il] > 0.f || conf.sysErr2Col[il] > 0.f) {
        return true;
      }
    }
    return false;
  }
}

/// Add configured systematic-error corrections to `sigma2Row` and `sigma2Col`.
/// ITS is a no-op.
template <o2::detectors::DetID::ID DetId>
void addSysErrors(int layerId, float& sigma2Row, float& sigma2Col)
{
  if constexpr (DetId == o2::detectors::DetID::ITS) {
    (void)layerId;
    (void)sigma2Row;
    (void)sigma2Col;
  } else {
    const auto& conf = o2::itsmft::tracking::TrackerParamRef<DetId>::get();
    sigma2Row += conf.sysErr2Row[layerId];
    sigma2Col += conf.sysErr2Col[layerId];
  }
}
} // namespace detail

constexpr float DefClusErrorRow = o2::itsmft::SegmentationAlpide::PitchRow * 0.5f;
constexpr float DefClusErrorCol = o2::itsmft::SegmentationAlpide::PitchCol * 0.5f;
constexpr float DefClusError2Row = DefClusErrorRow * DefClusErrorRow;
constexpr float DefClusError2Col = DefClusErrorCol * DefClusErrorCol;

void fillMatrixCache(o2::detectors::DetID::ID detId);

/// Decode detector geometry and covariance for one compact cluster.
template <o2::detectors::DetID::ID DetId>
o2::itsmft::tracking::ClusterDecodeResult decodeCluster(
  const CompClusterExt& c,
  o2::itsmft::tracking::BoundedPatternCursor& patterns,
  const TopologyDictionary* dict,
  bool applySysErrors);

template <class iterator, typename T>
o2::math_utils::Point3D<T> extractClusterData(const CompClusterExt& c, iterator& iter, const TopologyDictionary* dict, T& sig2Row, T& sig2Col, unsigned int* clusterSize = nullptr, o2::itsmft::tracking::ClusterShape* clusterShape = nullptr)
{
  auto pattID = c.getPatternID();
  sig2Row = DefClusError2Row;
  sig2Col = DefClusError2Col; // Default COG error (about half a pixel)
  const auto setShape = [clusterSize, clusterShape](const ClusterPattern& patt, unsigned int nPixels) {
    if (clusterSize != nullptr) {
      *clusterSize = nPixels;
    }
    if (clusterShape != nullptr) {
      *clusterShape = o2::itsmft::tracking::ClusterShape{
        nPixels, static_cast<uint16_t>(patt.getRowSpan()), static_cast<uint16_t>(patt.getColumnSpan())};
    }
  };
  if (pattID != CompCluster::InvalidPatternID) {
    sig2Row = dict->getErr2X(pattID);
    sig2Col = dict->getErr2Z(pattID);
    if (!dict->isGroup(pattID)) {
      setShape(dict->getPattern(pattID), dict->getNpixels(pattID));
      return dict->getClusterCoordinates<T>(c);
    }
    ClusterPattern patt(iter);
    setShape(patt, patt.getNPixels());
    return dict->getClusterCoordinates<T>(c, patt);
  }
  ClusterPattern patt(iter);
  setShape(patt, patt.getNPixels());
  return dict->getClusterCoordinates<T>(c, patt, false);
}

template <typename T>
struct ClusterDataDecodeResult {
  o2::math_utils::Point3D<T> coordinates{};
  T sig2Row{DefClusError2Row};
  T sig2Col{DefClusError2Col};
  o2::itsmft::tracking::ClusterShape shape{};
  o2::itsmft::tracking::ClusterDecodeError error{o2::itsmft::tracking::ClusterDecodeError::None};

  bool ok() const noexcept { return error == o2::itsmft::tracking::ClusterDecodeError::None; }
};

// Bounded counterpart: acquire pattern bytes only after validating the encoding.
template <typename T = float>
ClusterDataDecodeResult<T> extractClusterDataBounded(
  const CompClusterExt& c,
  o2::itsmft::tracking::BoundedPatternCursor& patterns,
  const TopologyDictionary* dict)
{
  ClusterDataDecodeResult<T> result;
  if (dict == nullptr) {
    result.error = o2::itsmft::tracking::ClusterDecodeError::MissingDictionary;
    return result;
  }

  const auto pattID = c.getPatternID();
  if (pattID != CompCluster::InvalidPatternID) {
    if (pattID >= dict->getSize()) {
      result.error = o2::itsmft::tracking::ClusterDecodeError::InvalidPatternId;
      return result;
    }
    result.sig2Row = dict->getErr2X(pattID);
    result.sig2Col = dict->getErr2Z(pattID);
    if (!dict->isGroup(pattID)) {
      const auto& pattern = dict->getPattern(pattID);
      result.shape = o2::itsmft::tracking::ClusterShape{
        static_cast<uint32_t>(dict->getNpixels(pattID)),
        static_cast<uint16_t>(pattern.getRowSpan()),
        static_cast<uint16_t>(pattern.getColumnSpan())};
      result.coordinates = dict->getClusterCoordinates<T>(c);
      return result;
    }
  }

  ClusterPattern pattern;
  result.error = patterns.acquirePattern(pattern);
  if (!result.ok()) {
    return result;
  }
  result.shape = o2::itsmft::tracking::ClusterShape{
    static_cast<uint32_t>(pattern.getNPixels()),
    static_cast<uint16_t>(pattern.getRowSpan()),
    static_cast<uint16_t>(pattern.getColumnSpan())};
  result.coordinates = dict->getClusterCoordinates<T>(c, pattern, pattID != CompCluster::InvalidPatternID);
  return result;
}

// Return coordinates as an array for TGeoMatrix callers.
template <class iterator, typename T>
std::array<T, 3> extractClusterDataA(const CompClusterExt& c, iterator& iter, const TopologyDictionary* dict, T& sig2Row, T& sig2Col)
{
  auto pattID = c.getPatternID();
  sig2Row = DefClusError2Row;
  sig2Col = DefClusError2Col; // Default COG error (about half a pixel)
  if (pattID != CompCluster::InvalidPatternID) {
    sig2Row = dict->getErr2X(pattID);
    sig2Col = dict->getErr2Z(pattID);
    if (!dict->isGroup(pattID)) {
      return dict->getClusterCoordinatesA<T>(c);
    }
    ClusterPattern patt(iter);
    return dict->getClusterCoordinatesA<T>(c, patt);
  }
  ClusterPattern patt(iter);
  return dict->getClusterCoordinatesA<T>(c, patt, false);
}

} // namespace o2::itsmft::ioutils

namespace o2::itsmft::tracking
{

class TimeFrame;

struct ClusterSourceInput {
  ClusterSourceId id{};
  o2::detectors::DetID::ID detector{o2::detectors::DetID::ITS};
  gsl::span<const o2::itsmft::CompClusterExt> clusters{};
  gsl::span<const unsigned char> patterns{};
  gsl::span<const o2::itsmft::ROFRecord> rofs{};
  const o2::itsmft::TopologyDictionary* dictionary{nullptr};
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels{nullptr};
  gsl::span<const LayerId> layerToSurface{};
  ROFTimingConfig timing{};
  const ClusterDecoder* decoder{nullptr};
  bool applySysErrors{true};
  RuntimeROFViews rofViews{};
};

enum class MultiSourceLoadError : uint8_t {
  None,
  NonDenseSourceIds,
  DuplicateSourceId,
  UnsupportedDetector,
  MissingDecoder,
  InvalidROFRange,
  InvalidLayerMapping,
  DetectorSurfaceMismatch,
  InconsistentDecoderMetadata,
  TimingError,
  SurfaceCatalogNotConfigured,
  SurfaceCatalogStale,
  MissingDictionary,
  TruncatedExplicitPattern,
  MalformedExplicitPattern,
  InvalidPatternId,
  InvalidSensor,
  InvalidDecodedLayer,
  GeometryUnavailable,
  OtherMalformedInput,
  TrailingPatternData,
  FrameNotConfigured
};

struct LoadSourcesResult {
  MultiSourceLoadError error{MultiSourceLoadError::None};
  ClusterSourceId source{};
  uint32_t rof{std::numeric_limits<uint32_t>::max()};
  uint32_t clusterIndex{std::numeric_limits<uint32_t>::max()};
  TimingBuildError timingDetail{TimingBuildError::None};
  bool ok() const noexcept { return error == MultiSourceLoadError::None; }
};

LoadSourcesResult loadSources(TimeFrame&, const SurfaceCatalogView&,
                              gsl::span<const ClusterSourceInput>,
                              const o2::InteractionRecord&,
                              std::vector<std::vector<uint32_t>>* externalIndicesBySurface = nullptr,
                              std::vector<std::vector<uint32_t>>* clusterSizesBySurface = nullptr);

/// Reset, decode, and normalize all sources into a configured TimeFrame.
/// A failed load leaves the TimeFrame empty.
LoadSourcesResult loadTimeFrameSources(TimeFrame&, gsl::span<const ClusterSourceInput>,
                                       SurfaceCatalogView, const o2::InteractionRecord&,
                                       std::vector<std::vector<uint32_t>>* externalIndicesBySurface = nullptr,
                                       std::vector<std::vector<uint32_t>>* clusterSizesBySurface = nullptr);

/// Convenience wrapper for a single detector source.
LoadSourcesResult loadTimeFrameSource(
  TimeFrame&, const ClusterDecoder&, const o2::InteractionRecord&, const ROFTimingConfig&,
  gsl::span<const itsmft::CompClusterExt>, gsl::span<const unsigned char>,
  gsl::span<const o2::itsmft::ROFRecord>, const itsmft::TopologyDictionary*,
  const dataformats::MCTruthContainer<MCCompLabel>*, o2::detectors::DetID::ID,
  gsl::span<const LayerId>, SurfaceCatalogView, bool applySysErrors = true,
  std::vector<std::vector<uint32_t>>* externalIndicesBySurface = nullptr,
  std::vector<std::vector<uint32_t>>* clusterSizesBySurface = nullptr);

#ifndef GPUCA_GPUCODE
class RecoverableLoadFailure final : public std::runtime_error
{
 public:
  explicit RecoverableLoadFailure(const LoadSourcesResult& result);
  MultiSourceLoadError error() const noexcept { return mResult.error; }
  const LoadSourcesResult& result() const noexcept { return mResult; }

 private:
  LoadSourcesResult mResult;
};

enum class TimeFrameLoadFailureReason : uint8_t {
  DictionaryNotConfigured,
  NonUniformROFTiming,
  ZeroROFCount,
  LoadSourcesFailure
};

class TimeFrameLoadException final : public std::runtime_error
{
 public:
  TimeFrameLoadException(TimeFrameLoadFailureReason, std::string);
  explicit TimeFrameLoadException(const LoadSourcesResult&);
  TimeFrameLoadFailureReason reason() const noexcept { return mReason; }
  const LoadSourcesResult& loadResult() const noexcept { return mLoadResult; }

 private:
  TimeFrameLoadFailureReason mReason;
  LoadSourcesResult mLoadResult{};
};

bool isRecoverableLoadError(MultiSourceLoadError, TimingBuildError) noexcept;
#endif

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_IOUTILS_H_ */
