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

#include <cstddef>
#include <cstdint>

#ifndef GPUCA_GPUCODE
#include <format>
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
#include "ITSMFTTracking/GlobalMeasurement.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/ROFViews.h"
#include "ITSMFTTracking/SurfaceDescriptor.h"
#include "ITSMFTTracking/SurfaceMeasurement.h"
#include "ITSMFTTracking/SurfaceTiming.h"
#include "MathUtils/Cartesian.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2::itsmft::tracking
{

// Host-side facts produced by compact-cluster and geometry decoding.
struct DecodedCluster {
  GlobalPoint3F global{};
  // ITS geometry supplies its cylindrical tracking frame here. Disk
  // projection uses global coordinates directly.
  SurfaceFramePoint cylinderFrame{};
  // ALPIDE local row/column covariance. The detector projection determines
  // which normalized axes these values describe.
  SurfaceCovariance2F rowColumnCovariance{};
  uint32_t nPixels{0};
  int layer{-1};
};

} // namespace o2::itsmft::tracking

namespace o2::itsmft::ioutils
{

constexpr float DefClusErrorRow = o2::itsmft::SegmentationAlpide::PitchRow * 0.5f;
constexpr float DefClusErrorCol = o2::itsmft::SegmentationAlpide::PitchCol * 0.5f;
constexpr float DefClusError2Row = DefClusErrorRow * DefClusErrorRow;
constexpr float DefClusError2Col = DefClusErrorCol * DefClusErrorCol;

template <typename T>
struct ClusterData {
  o2::math_utils::Point3D<T> coordinates{};
  T sig2Row{DefClusError2Row};
  T sig2Col{DefClusError2Col};
  uint32_t nPixels{0};
};

// Decode using dictionary coordinates, errors and pixel counts. Grouped and
// explicit patterns require their actual bitmap; the group entry is representative.
// As in ITS tracking, the explicit-pattern stream is assumed to be valid.
template <typename T = float>
ClusterData<T> extractClusterData(
  const CompClusterExt& c,
  gsl::span<const unsigned char>::iterator& patterns,
  const TopologyDictionary* dict)
{
  ClusterData<T> result;
  if (dict == nullptr) {
    throw std::runtime_error("Cluster dictionary is not available");
  }

  const auto pattID = c.getPatternID();
  if (pattID != CompCluster::InvalidPatternID) {
    if (pattID >= dict->getSize()) {
      throw std::runtime_error("Cluster pattern ID is outside the topology dictionary");
    }
    result.sig2Row = dict->getErr2X(pattID);
    result.sig2Col = dict->getErr2Z(pattID);
    if (!dict->isGroup(pattID)) {
      result.nPixels = static_cast<uint32_t>(dict->getNpixels(pattID));
      result.coordinates = dict->getClusterCoordinates<T>(c);
      return result;
    }
  }

  const o2::itsmft::ClusterPattern pattern{patterns};
  result.nPixels = static_cast<uint32_t>(pattern.getNPixels());
  result.coordinates = TopologyDictionary::getClusterCoordinates<T>(c, pattern, pattID != CompCluster::InvalidPatternID);
  return result;
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
  bool applySysErrors{true};
  RuntimeROFViews rofViews{};
};

/// Reset, decode, and normalize all sources into a configured TimeFrame.
/// Invalid input throws. On failure, the caller must reset the frame before
/// reuse; partially loaded data must not be published.
void loadTimeFrameSources(TimeFrame&, gsl::span<const ClusterSourceInput>,
                          SurfaceCatalogView, const o2::InteractionRecord&,
                          std::vector<std::vector<uint32_t>>* externalIndicesBySurface = nullptr,
                          std::vector<std::vector<uint32_t>>* clusterSizesBySurface = nullptr);

namespace detail
{
void prepareSources(TimeFrame&, const SurfaceCatalogView&, gsl::span<const ClusterSourceInput>,
                    std::vector<std::vector<uint32_t>>*, std::vector<std::vector<uint32_t>>*, bool requireCompleteMapping = false);
void validateSource(const ClusterSourceInput&, const o2::InteractionRecord&);
void appendCluster(TimeFrame&, const SurfaceCatalogView&, const ClusterSourceInput&, const DecodedCluster&,
                   uint32_t, uint32_t, std::vector<std::vector<uint32_t>>&, std::vector<std::vector<uint32_t>>&);
void bindSourceROFNavigation(TimeFrame&, const ClusterSourceInput&, const std::vector<std::vector<int>>&);

// Internal loading loop; geometry decoding and synthetic fixtures share the
// same stream consumption, diagnostics and measurement insertion.
template <typename Decode>
void loadDecodedSource(TimeFrame& frame, const SurfaceCatalogView& catalog, const ClusterSourceInput& src,
                       const Decode& decode, std::vector<std::vector<uint32_t>>& externalIndices,
                       std::vector<std::vector<uint32_t>>& clusterSizes)
{
  std::vector<std::vector<int>> boundaries(src.layerToSurface.size(), std::vector<int>(src.rofs.size() + 1, 0));
  auto patterns = src.patterns.begin();
  for (uint32_t r = 0; r < src.rofs.size(); ++r) {
    const auto& rof = src.rofs[r];
    const auto firstEntry = rof.getFirstEntry();
    const auto nEntries = rof.getNEntries();
    for (int32_t clusterId = firstEntry; clusterId < firstEntry + nEntries; ++clusterId) {
      const auto& cluster = src.clusters[clusterId];
      const auto externalIndex = static_cast<uint32_t>(clusterId);
      DecodedCluster decoded;
      try {
        decoded = decode(cluster, patterns);
      } catch (const std::runtime_error& error) {
        throw std::runtime_error(std::format("Cluster decoding failed: source={} rof={} clusterIndex={}: {}",
                                             src.id.value(), r, externalIndex, error.what()));
      }
      appendCluster(frame, catalog, src, decoded, r, externalIndex, externalIndices, clusterSizes);
    }
    for (size_t layer = 0; layer < src.layerToSurface.size(); ++layer) {
      boundaries[layer][r + 1] = static_cast<int>(externalIndices[src.layerToSurface[layer].value()].size());
    }
  }
  if (patterns != src.patterns.end()) {
    throw std::runtime_error(std::format("Trailing cluster pattern data source={} rof={} clusterIndex={}", src.id.value(), static_cast<uint32_t>(src.rofs.size()), static_cast<uint32_t>(src.clusters.size())));
  }
  bindSourceROFNavigation(frame, src, boundaries);
}
} // namespace detail

} // namespace o2::itsmft::tracking

#endif /* ALICEO2_ITSMFT_TRACKING_IOUTILS_H_ */
