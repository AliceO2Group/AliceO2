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

#include "ITSMFTTracking/IOUtils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <vector>

#include "ITSMFTTracking/TimeFrame.h"
#include "Framework/Logger.h"
#include "GPUCommonMath.h"
#include "ITSBase/GeometryTGeo.h"
#include "MFTBase/GeometryTGeo.h"
#include "MathUtils/Utils.h"

namespace
{

using o2::itsmft::ioutils::detail::addSysErrors;
using o2::itsmft::ioutils::detail::shouldApplySysErrors;

template <o2::detectors::DetID::ID DetId, typename GeomT>
o2::itsmft::tracking::ClusterDecodeResult decodeClusterBounded(
  GeomT* geom, const o2::itsmft::CompClusterExt& cluster,
  o2::itsmft::tracking::BoundedPatternCursor& patterns,
  const o2::itsmft::TopologyDictionary* dict, bool applySysErrors)
{
  using o2::itsmft::tracking::ClusterDecodeError;
  o2::itsmft::tracking::ClusterDecodeResult result;
  if (dict == nullptr) {
    result.error = ClusterDecodeError::MissingDictionary;
    return result;
  }
  if (geom == nullptr) {
    result.error = ClusterDecodeError::GeometryUnavailable;
    return result;
  }

  const auto sensorID = cluster.getSensorID();
  if (!o2::itsmft::ioutils::detail::isSensorInGeometry(sensorID, geom->getSize())) {
    result.error = ClusterDecodeError::InvalidSensor;
    return result;
  }
  const int layer = geom->getLayer(sensorID);
  if (!o2::itsmft::ioutils::detail::isLayerInDetector(layer, o2::itsmft::tracking::TrackerParamRef<DetId>::nLayers())) {
    result.error = ClusterDecodeError::InvalidLayer;
    return result;
  }

  const auto clusterData = o2::itsmft::ioutils::extractClusterDataBounded(cluster, patterns, dict);
  if (!clusterData.ok()) {
    result.error = clusterData.error;
    return result;
  }
  float sigma2Row = clusterData.sig2Row;
  float sigma2Col = clusterData.sig2Col;
  if (applySysErrors && shouldApplySysErrors<DetId>()) {
    addSysErrors<DetId>(layer, sigma2Row, sigma2Col);
  }

  if constexpr (DetId == o2::detectors::DetID::ITS) {
    const auto trkXYZ = geom->getMatrixT2L(sensorID) ^ clusterData.coordinates;
    const auto gloXYZ = geom->getMatrixL2G(sensorID) * clusterData.coordinates;
    result.decoded = {{gloXYZ.x(), gloXYZ.y(), gloXYZ.z()},
                      {trkXYZ.x(), trkXYZ.y(), trkXYZ.z(), geom->getSensorRefAlpha(sensorID)},
                      {sigma2Row, 0.f, sigma2Col},
                      clusterData.shape,
                      layer};
  } else {
    if (!geom->getCacheL2G().isFilled() || geom->getCacheL2G().getSize() <= sensorID) {
      result.error = ClusterDecodeError::GeometryUnavailable;
      return result;
    }
    const auto gloXYZ = geom->getMatrixL2G(sensorID) * clusterData.coordinates;
    result.decoded = {{gloXYZ.x(), gloXYZ.y(), gloXYZ.z()}, {}, {sigma2Row, 0.f, sigma2Col}, clusterData.shape, layer};
  }
  return result;
}

} // namespace

namespace o2::itsmft::ioutils
{

void fillMatrixCache(o2::detectors::DetID::ID detId)
{
  const auto mask = o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G);
  if (detId == o2::detectors::DetID::ITS) {
    o2::its::GeometryTGeo::Instance()->fillMatrixCache(mask);
  } else if (detId == o2::detectors::DetID::MFT) {
    o2::mft::GeometryTGeo::Instance()->fillMatrixCache(mask);
  } else {
    LOGP(fatal, "Unsupported detector id {} in fillMatrixCache", static_cast<int>(detId));
  }
}

template <o2::detectors::DetID::ID DetId>
o2::itsmft::tracking::ClusterDecodeResult decodeCluster(
  const CompClusterExt& cluster, o2::itsmft::tracking::BoundedPatternCursor& patterns,
  const TopologyDictionary* dict, bool applySysErrors)
{
  if constexpr (DetId == o2::detectors::DetID::ITS) {
    return decodeClusterBounded<DetId>(o2::its::GeometryTGeo::Instance(), cluster, patterns, dict, applySysErrors);
  } else {
    return decodeClusterBounded<DetId>(o2::mft::GeometryTGeo::Instance(), cluster, patterns, dict, applySysErrors);
  }
}

template o2::itsmft::tracking::ClusterDecodeResult decodeCluster<o2::detectors::DetID::ITS>(
  const CompClusterExt&, o2::itsmft::tracking::BoundedPatternCursor&, const TopologyDictionary*, bool);
template o2::itsmft::tracking::ClusterDecodeResult decodeCluster<o2::detectors::DetID::MFT>(
  const CompClusterExt&, o2::itsmft::tracking::BoundedPatternCursor&, const TopologyDictionary*, bool);

} // namespace o2::itsmft::ioutils

namespace o2::itsmft::tracking
{

namespace
{
class FailedTimeFrameLoadGuard
{
 public:
  explicit FailedTimeFrameLoadGuard(TimeFrame& frame) noexcept : mFrame{&frame} {}
  ~FailedTimeFrameLoadGuard()
  {
    if (mFrame != nullptr) {
      mFrame->resetTimeFrame();
    }
  }
  void release() noexcept { mFrame = nullptr; }

 private:
  TimeFrame* mFrame;
};

void clearFrameAndSidecars(TimeFrame& frame,
                           std::vector<std::vector<uint32_t>>* externalIndicesBySurface,
                           std::vector<std::vector<uint32_t>>* clusterSizesBySurface) noexcept
{
  frame.resetTimeFrame();
  if (externalIndicesBySurface != nullptr) {
    externalIndicesBySurface->clear();
  }
  if (clusterSizesBySurface != nullptr) {
    clusterSizesBySurface->clear();
  }
}

LoadSourcesResult decodeSources(TimeFrame& frame, const SurfaceCatalogView& catalog,
                                gsl::span<const ClusterSourceInput> sources,
                                const o2::InteractionRecord& origin,
                                std::vector<std::vector<uint32_t>>* externalIndicesBySurface,
                                std::vector<std::vector<uint32_t>>* clusterSizesBySurface);
} // namespace

LoadSourcesResult loadTimeFrameSources(TimeFrame& frame, gsl::span<const ClusterSourceInput> sources,
                                       SurfaceCatalogView catalog, const o2::InteractionRecord& origin,
                                       std::vector<std::vector<uint32_t>>* externalIndicesBySurface,
                                       std::vector<std::vector<uint32_t>>* clusterSizesBySurface)
{
  clearFrameAndSidecars(frame, externalIndicesBySurface, clusterSizesBySurface);
  if (!frame.isConfigured()) {
    return {MultiSourceLoadError::FrameNotConfigured};
  }
  if (sources.empty()) {
    return {MultiSourceLoadError::OtherMalformedInput};
  }
  FailedTimeFrameLoadGuard failedLoad{frame};
  std::vector<std::vector<uint32_t>> loadedExternalIndices;
  std::vector<std::vector<uint32_t>> loadedClusterSizes;
  const auto loadResult = decodeSources(frame, catalog, sources, origin,
                                        &loadedExternalIndices, &loadedClusterSizes);
  if (!loadResult.ok()) {
    return loadResult;
  }

  const auto& layout = frame.getLayout();
  if (layout.empty()) {
    return {MultiSourceLoadError::FrameNotConfigured};
  }
  std::array<bool, MaxLayoutSurfaces> configuredSurfaces{};
  for (std::size_t position = 0; position < layout.size(); ++position) {
    configuredSurfaces[position] = true;
  }
  std::array<bool, MaxLayoutSurfaces> mappedSurfaces{};
  for (const auto& source : sources) {
    for (const auto surface : source.layerToSurface) {
      if (!surface.isValid() || surface.value() >= MaxLayoutSurfaces ||
          mappedSurfaces[surface.value()] || !configuredSurfaces[surface.value()]) {
        return {MultiSourceLoadError::InvalidLayerMapping, source.id};
      }
      if (catalog.getSurface(surface).detectorId != static_cast<uint8_t>(source.detector)) {
        return {MultiSourceLoadError::DetectorSurfaceMismatch, source.id};
      }
      mappedSurfaces[surface.value()] = true;
    }
  }
  if (mappedSurfaces != configuredSurfaces) {
    // Attribute an omitted surface only when one source owns its detector.
    for (uint16_t position = 0; position < layout.size(); ++position) {
      const auto surface = LayerId{position};
      if (mappedSurfaces[surface.value()]) {
        continue;
      }
      ClusterSourceId owner;
      for (const auto& source : sources) {
        if (static_cast<uint8_t>(source.detector) != catalog.getSurface(surface).detectorId) {
          continue;
        }
        if (owner.isValid()) {
          return {MultiSourceLoadError::InvalidLayerMapping};
        }
        owner = source.id;
      }
      return {MultiSourceLoadError::InvalidLayerMapping, owner};
    }
    return {MultiSourceLoadError::InvalidLayerMapping};
  }

  frame.setROFViews(sources.front().rofViews);
  for (uint16_t position = 0; position < layout.size(); ++position) {
    const auto surface = LayerId{position};
    const ClusterSourceInput* owner = nullptr;
    uint16_t localLayer = 0;
    for (const auto& source : sources) {
      const auto it = std::find(source.layerToSurface.begin(), source.layerToSurface.end(), surface);
      if (it == source.layerToSurface.end()) {
        continue;
      }
      if (owner != nullptr) {
        return {MultiSourceLoadError::InvalidLayerMapping, source.id};
      }
      owner = &source;
      localLayer = static_cast<uint16_t>(std::distance(source.layerToSurface.begin(), it));
    }
    if (owner == nullptr) {
      return {MultiSourceLoadError::InvalidLayerMapping};
    }

    const auto globals = frame.getGlobalMeasurements(surface);
    std::vector<int> boundaries;
    boundaries.assign(owner->rofs.size() + 1, 0);
    std::size_t measurement = 0;
    for (std::size_t rof = 0; rof < owner->rofs.size(); ++rof) {
      const auto firstEntry = static_cast<uint32_t>(owner->rofs[rof].getFirstEntry());
      const auto endEntry = firstEntry + static_cast<uint32_t>(owner->rofs[rof].getNEntries());
      while (measurement < globals.size()) {
        const auto clusterId = globals[measurement].clusterId;
        if (surface.value() >= loadedExternalIndices.size() ||
            clusterId >= loadedExternalIndices[surface.value()].size()) {
          return {MultiSourceLoadError::InconsistentDecoderMetadata, owner->id,
                  static_cast<uint32_t>(rof), clusterId};
        }
        const auto externalIndex = loadedExternalIndices[surface.value()][clusterId];
        if (externalIndex >= endEntry) {
          break;
        }
        if (externalIndex < firstEntry) {
          return {MultiSourceLoadError::InconsistentDecoderMetadata, owner->id,
                  static_cast<uint32_t>(rof), externalIndex};
        }
        ++measurement;
      }
      boundaries[rof + 1] = static_cast<int>(measurement);
    }
    if (measurement != globals.size()) {
      return {MultiSourceLoadError::InconsistentDecoderMetadata, owner->id};
    }
    frame.setROFNavigation(position, boundaries, owner->rofViews, localLayer);
  }
  if (externalIndicesBySurface != nullptr) {
    *externalIndicesBySurface = std::move(loadedExternalIndices);
  }
  if (clusterSizesBySurface != nullptr) {
    *clusterSizesBySurface = std::move(loadedClusterSizes);
  }
  failedLoad.release();
  return {};
}

LoadSourcesResult loadTimeFrameSource(
  TimeFrame& frame,
  const ClusterDecoder& decoder,
  const o2::InteractionRecord& origin,
  const ROFTimingConfig& timing,
  gsl::span<const itsmft::CompClusterExt> clusters,
  gsl::span<const unsigned char> patterns,
  gsl::span<const o2::itsmft::ROFRecord> rofs,
  const itsmft::TopologyDictionary* dictionary,
  const dataformats::MCTruthContainer<MCCompLabel>* labels,
  o2::detectors::DetID::ID detector,
  gsl::span<const LayerId> layerToSurface,
  SurfaceCatalogView catalog,
  bool applySysErrors,
  std::vector<std::vector<uint32_t>>* externalIndicesBySurface,
  std::vector<std::vector<uint32_t>>* clusterSizesBySurface)
{
  constexpr ClusterSourceId sourceId{0};
  if (detector != o2::detectors::DetID::ITS && detector != o2::detectors::DetID::MFT) {
    clearFrameAndSidecars(frame, externalIndicesBySurface, clusterSizesBySurface);
    return {MultiSourceLoadError::UnsupportedDetector, sourceId};
  }
  if (catalog.surfaces == nullptr || catalog.nSurfaces == 0) {
    clearFrameAndSidecars(frame, externalIndicesBySurface, clusterSizesBySurface);
    return {MultiSourceLoadError::SurfaceCatalogNotConfigured, sourceId};
  }
  ClusterSourceInput source;
  source.id = sourceId;
  source.detector = detector;
  source.clusters = clusters;
  source.patterns = patterns;
  source.rofs = rofs;
  source.dictionary = dictionary;
  source.labels = labels;
  source.layerToSurface = layerToSurface;
  source.timing = timing;
  source.decoder = &decoder;
  source.applySysErrors = applySysErrors;
  source.rofViews = frame.getROFViews();
  return loadTimeFrameSources(frame, gsl::span<const ClusterSourceInput>{&source, 1}, catalog, origin,
                              externalIndicesBySurface, clusterSizesBySurface);
}

namespace
{
bool isSupportedDetector(o2::detectors::DetID::ID det) noexcept
{
  return det == o2::detectors::DetID::ITS || det == o2::detectors::DetID::MFT;
}

bool covariance2DIsPositiveSemidefinite(float varianceFirst, float covariance,
                                        float varianceSecond) noexcept
{
  if (!o2::gpu::GPUCommonMath::Finite(varianceFirst) || !o2::gpu::GPUCommonMath::Finite(covariance) ||
      !o2::gpu::GPUCommonMath::Finite(varianceSecond) || varianceFirst < 0.f || varianceSecond < 0.f) {
    return false;
  }
  const double diagonalProduct = static_cast<double>(varianceFirst) * varianceSecond;
  const double covarianceSquared = static_cast<double>(covariance) * covariance;
  const double tolerance = 16. * std::numeric_limits<float>::epsilon() *
                           std::max(diagonalProduct, covarianceSquared);
  return diagonalProduct - covarianceSquared >= -tolerance;
}

bool globalCovarianceIsPositiveSemidefinite(const GlobalCovariance3F& covariance) noexcept
{
  const float xx = covariance[GlobalMeasurement::XX];
  const float xy = covariance[GlobalMeasurement::XY];
  const float xz = covariance[GlobalMeasurement::XZ];
  const float yy = covariance[GlobalMeasurement::YY];
  const float yz = covariance[GlobalMeasurement::YZ];
  const float zz = covariance[GlobalMeasurement::ZZ];
  if (!covariance2DIsPositiveSemidefinite(xx, xy, yy) ||
      !covariance2DIsPositiveSemidefinite(xx, xz, zz) ||
      !covariance2DIsPositiveSemidefinite(yy, yz, zz)) {
    return false;
  }
  const double determinant =
    static_cast<double>(xx) * yy * zz + 2. * static_cast<double>(xy) * xz * yz -
    static_cast<double>(xx) * yz * yz - static_cast<double>(yy) * xz * xz -
    static_cast<double>(zz) * xy * xy;
  const double scale = std::max({std::abs(static_cast<double>(xx) * yy * zz),
                                 std::abs(2. * static_cast<double>(xy) * xz * yz),
                                 std::abs(static_cast<double>(xx) * yz * yz),
                                 std::abs(static_cast<double>(yy) * xz * xz),
                                 std::abs(static_cast<double>(zz) * xy * xy)});
  return o2::gpu::GPUCommonMath::Finite(static_cast<float>(determinant)) &&
         determinant >= -32. * std::numeric_limits<float>::epsilon() * scale;
}

bool decodedMeasurementIsValid(const GlobalMeasurement& global,
                               const SurfaceMeasurement& local) noexcept
{
  return o2::gpu::GPUCommonMath::Finite(global.x) && o2::gpu::GPUCommonMath::Finite(global.y) &&
         o2::gpu::GPUCommonMath::Finite(global.z) &&
         globalCovarianceIsPositiveSemidefinite(global.covariance) &&
         o2::gpu::GPUCommonMath::Finite(local.frame.q) && o2::gpu::GPUCommonMath::Finite(local.frame.u) &&
         o2::gpu::GPUCommonMath::Finite(local.frame.v) && o2::gpu::GPUCommonMath::Finite(local.frame.frameAngle) &&
         covariance2DIsPositiveSemidefinite(local.covariance.uu,
                                            local.covariance.uv,
                                            local.covariance.vv);
}

MultiSourceLoadError mapDecodeError(ClusterDecodeError error) noexcept
{
  switch (error) {
    case ClusterDecodeError::None:
      return MultiSourceLoadError::None;
    case ClusterDecodeError::MissingDictionary:
      return MultiSourceLoadError::MissingDictionary;
    case ClusterDecodeError::TruncatedExplicitPattern:
      return MultiSourceLoadError::TruncatedExplicitPattern;
    case ClusterDecodeError::MalformedExplicitPattern:
      return MultiSourceLoadError::MalformedExplicitPattern;
    case ClusterDecodeError::InvalidPatternId:
      return MultiSourceLoadError::InvalidPatternId;
    case ClusterDecodeError::InvalidSensor:
      return MultiSourceLoadError::InvalidSensor;
    case ClusterDecodeError::InvalidLayer:
      return MultiSourceLoadError::InvalidDecodedLayer;
    case ClusterDecodeError::GeometryUnavailable:
      return MultiSourceLoadError::GeometryUnavailable;
    case ClusterDecodeError::OtherMalformedInput:
      return MultiSourceLoadError::OtherMalformedInput;
  }
  return MultiSourceLoadError::OtherMalformedInput;
}
} // namespace

namespace
{
LoadSourcesResult decodeSources(TimeFrame& frame,
                                const SurfaceCatalogView& catalog,
                                gsl::span<const ClusterSourceInput> sources,
                                const o2::InteractionRecord& origin,
                                std::vector<std::vector<uint32_t>>* externalIndicesBySurface,
                                std::vector<std::vector<uint32_t>>* clusterSizesBySurface)
{
  const auto nSources = static_cast<uint32_t>(sources.size());

  std::vector<bool> seen(nSources, false);
  std::vector<ClusterSourceId> sourceBySurface(catalog.nSurfaces, ClusterSourceId::invalid());
  for (const auto& src : sources) {
    if (!src.id.isValid() || src.id.value() >= nSources) {
      return {MultiSourceLoadError::NonDenseSourceIds, src.id};
    }
    if (seen[src.id.value()]) {
      return {MultiSourceLoadError::DuplicateSourceId, src.id};
    }
    seen[src.id.value()] = true;
    if (!isSupportedDetector(src.detector)) {
      return {MultiSourceLoadError::UnsupportedDetector, src.id};
    }
    if (src.decoder == nullptr) {
      return {MultiSourceLoadError::MissingDecoder, src.id};
    }
    if (!src.clusters.empty() && src.dictionary == nullptr) {
      return {MultiSourceLoadError::MissingDictionary, src.id, 0, 0};
    }
    for (const auto surface : src.layerToSurface) {
      if (!surface.isValid() || surface.value() >= catalog.nSurfaces) {
        return {MultiSourceLoadError::InvalidLayerMapping, src.id};
      }
      if (sourceBySurface[surface.value()].isValid()) {
        return {MultiSourceLoadError::InvalidLayerMapping, src.id};
      }
      if (catalog.getSurface(surface).detectorId != static_cast<uint8_t>(src.detector)) {
        return {MultiSourceLoadError::DetectorSurfaceMismatch, src.id};
      }
      sourceBySurface[surface.value()] = src.id;
    }
  }

  std::vector<std::vector<uint32_t>> perSurfaceClusterSizes(catalog.nSurfaces);
  std::vector<std::vector<uint32_t>> stagedExternalIndices(catalog.nSurfaces);
  bool hasMCInformation = false;

  for (const auto& src : sources) {
    hasMCInformation |= src.labels != nullptr;

    int64_t expectedNext = 0;
    for (uint32_t r = 0; r < src.rofs.size(); ++r) {
      const auto& rof = src.rofs[r];
      const int64_t first = rof.getFirstEntry();
      const int64_t n = rof.getNEntries();
      if (n < 0 || first != expectedNext) {
        return {MultiSourceLoadError::InvalidROFRange, src.id, r};
      }
      expectedNext = first + n;
      if (expectedNext > static_cast<int64_t>(src.clusters.size())) {
        return {MultiSourceLoadError::InvalidROFRange, src.id, r};
      }
    }
    if (expectedNext != static_cast<int64_t>(src.clusters.size())) {
      return {MultiSourceLoadError::InvalidROFRange, src.id, static_cast<uint32_t>(src.rofs.size())};
    }

    for (uint32_t r = 0; r < src.rofs.size(); ++r) {
      const auto built = computeROFIntervalBC(src.rofs[r].getBCData(), origin, src.timing, r);
      if (!built.ok()) {
        return LoadSourcesResult{.error = MultiSourceLoadError::TimingError, .source = src.id, .rof = r, .timingDetail = built.error};
      }
    }

    src.decoder->prepare();
    BoundedPatternCursor patterns{src.patterns};
    for (uint32_t r = 0; r < src.rofs.size(); ++r) {
      const auto& rof = src.rofs[r];
      const auto firstEntry = rof.getFirstEntry();
      const auto nEntries = rof.getNEntries();
      for (int32_t clusterId = firstEntry; clusterId < firstEntry + nEntries; ++clusterId) {
        const auto& cluster = src.clusters[clusterId];
        const auto externalIndex = static_cast<uint32_t>(clusterId);
        const auto decodeResult = src.decoder->decode(cluster, patterns, src.dictionary,
                                                      externalIndex, src.applySysErrors);
        if (!decodeResult.ok()) {
          return {mapDecodeError(decodeResult.error), src.id, r, externalIndex};
        }
        const auto& decoded = decodeResult.decoded;
        if (decoded.layer < 0 || static_cast<size_t>(decoded.layer) >= src.layerToSurface.size()) {
          return {MultiSourceLoadError::InvalidLayerMapping, src.id, r, externalIndex};
        }
        const auto expectedSurface = src.layerToSurface[decoded.layer];
        if (!expectedSurface.isValid() || expectedSurface.value() >= catalog.nSurfaces) {
          return {MultiSourceLoadError::InvalidLayerMapping, src.id, r, externalIndex};
        }
        const auto& surfaceDescriptor = catalog.getSurface(expectedSurface);
        if (surfaceDescriptor.detectorId != static_cast<uint8_t>(src.detector)) {
          return {MultiSourceLoadError::DetectorSurfaceMismatch, src.id, r, externalIndex};
        }
        const auto localClusterId = static_cast<uint32_t>(frame.getGlobalMeasurements(expectedSurface).size());
        GlobalMeasurement global;
        SurfaceMeasurement measurement;
        if (surfaceDescriptor.kind == SurfaceKind::Cylinder) {
          global = makeCylinderGlobalMeasurement(decoded, localClusterId);
          measurement = makeCylinderSurfaceMeasurement(decoded);
        } else {
          global = makeDiskGlobalMeasurement(decoded, localClusterId);
          measurement = makeDiskSurfaceMeasurement(decoded);
        }
        if (!decodedMeasurementIsValid(global, measurement)) {
          return {MultiSourceLoadError::OtherMalformedInput, src.id, r, externalIndex};
        }
        global.x -= frame.getBeamX();
        global.y -= frame.getBeamY();
        global.radius = std::hypot(global.x, global.y);
        global.phi = o2::its::math_utils::computePhi(global.x, global.y);
        if (src.labels != nullptr) {
          frame.addMeasurement(expectedSurface, global, measurement, src.labels->getLabels(externalIndex));
        } else {
          frame.addMeasurement(expectedSurface, global, measurement);
        }
        perSurfaceClusterSizes[expectedSurface.value()].push_back(decoded.shape.nPixels);
        stagedExternalIndices[expectedSurface.value()].push_back(externalIndex);
      }
    }
    if (!patterns.empty()) {
      return {MultiSourceLoadError::TrailingPatternData, src.id,
              static_cast<uint32_t>(src.rofs.size()), static_cast<uint32_t>(src.clusters.size())};
    }
  }

  frame.setHasMCInformation(hasMCInformation);
  if (externalIndicesBySurface != nullptr) {
    *externalIndicesBySurface = std::move(stagedExternalIndices);
  }
  if (clusterSizesBySurface != nullptr) {
    *clusterSizesBySurface = std::move(perSurfaceClusterSizes);
  }
  return {};
}
} // namespace

LoadSourcesResult loadSources(TimeFrame& frame, const SurfaceCatalogView& catalog,
                              gsl::span<const ClusterSourceInput> sources,
                              const o2::InteractionRecord& origin,
                              std::vector<std::vector<uint32_t>>* externalIndicesBySurface,
                              std::vector<std::vector<uint32_t>>* clusterSizesBySurface)
{
  clearFrameAndSidecars(frame, externalIndicesBySurface, clusterSizesBySurface);
  if (!frame.isConfigured()) {
    return {MultiSourceLoadError::FrameNotConfigured};
  }
  FailedTimeFrameLoadGuard failedLoad{frame};
  const auto result = decodeSources(frame, catalog, sources, origin,
                                    externalIndicesBySurface, clusterSizesBySurface);
  if (result.ok()) {
    failedLoad.release();
  }
  return result;
}

namespace
{
std::string formatLoadSourcesResult(const char* label, const LoadSourcesResult& result)
{
  return std::format("{}: error={} source={} rof={} clusterIndex={} timingDetail={}",
                     label, static_cast<int>(result.error), result.source.value(),
                     result.rof, result.clusterIndex, static_cast<int>(result.timingDetail));
}
} // namespace

RecoverableLoadFailure::RecoverableLoadFailure(const LoadSourcesResult& result)
  : std::runtime_error(formatLoadSourcesResult("TimeFrame loading boundary: recoverable data failure", result)),
    mResult(result)
{
}

TimeFrameLoadException::TimeFrameLoadException(TimeFrameLoadFailureReason reason, std::string message)
  : std::runtime_error(std::move(message)), mReason(reason)
{
}

TimeFrameLoadException::TimeFrameLoadException(const LoadSourcesResult& result)
  : std::runtime_error(formatLoadSourcesResult("TimeFrame loading boundary: structural failure", result)),
    mReason(TimeFrameLoadFailureReason::LoadSourcesFailure),
    mLoadResult(result)
{
}

bool isRecoverableLoadError(MultiSourceLoadError error, TimingBuildError timingDetail) noexcept
{
  switch (error) {
    case MultiSourceLoadError::InvalidROFRange:
    case MultiSourceLoadError::TruncatedExplicitPattern:
    case MultiSourceLoadError::MalformedExplicitPattern:
    case MultiSourceLoadError::InvalidPatternId:
    case MultiSourceLoadError::InvalidSensor:
    case MultiSourceLoadError::InvalidDecodedLayer:
    case MultiSourceLoadError::OtherMalformedInput:
    case MultiSourceLoadError::TrailingPatternData:
      return true;
    case MultiSourceLoadError::TimingError:
      return timingDetail == TimingBuildError::Overflow;
    case MultiSourceLoadError::None:
    case MultiSourceLoadError::NonDenseSourceIds:
    case MultiSourceLoadError::DuplicateSourceId:
    case MultiSourceLoadError::UnsupportedDetector:
    case MultiSourceLoadError::MissingDecoder:
    case MultiSourceLoadError::InvalidLayerMapping:
    case MultiSourceLoadError::DetectorSurfaceMismatch:
    case MultiSourceLoadError::InconsistentDecoderMetadata:
    case MultiSourceLoadError::SurfaceCatalogNotConfigured:
    case MultiSourceLoadError::SurfaceCatalogStale:
    case MultiSourceLoadError::MissingDictionary:
    case MultiSourceLoadError::GeometryUnavailable:
    case MultiSourceLoadError::FrameNotConfigured:
      return false;
  }
  return false;
}

} // namespace o2::itsmft::tracking
