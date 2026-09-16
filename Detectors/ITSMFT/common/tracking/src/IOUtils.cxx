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
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "ITSMFTTracking/TimeFrame.h"
#include "GPUCommonMath.h"
#include "ITSBase/GeometryTGeo.h"
#include "MFTBase/GeometryTGeo.h"
#include "MathUtils/Utils.h"

namespace
{

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

template <o2::detectors::DetID::ID DetId, typename GeomT>
o2::itsmft::tracking::DecodedCluster decodeCluster(
  GeomT* geom, const o2::itsmft::CompClusterExt& cluster,
  gsl::span<const unsigned char>::iterator& patterns,
  const o2::itsmft::TopologyDictionary* dict, bool applySysErrors)
{
  o2::itsmft::tracking::DecodedCluster result;
  if (dict == nullptr) {
    throw std::runtime_error("Cluster dictionary is not available");
  }
  if (geom == nullptr) {
    throw std::runtime_error("Cluster geometry is not available");
  }

  const auto sensorID = cluster.getSensorID();
  if (sensorID >= geom->getSize()) {
    throw std::runtime_error("Cluster sensor ID is outside the detector geometry");
  }
  const int layer = geom->getLayer(sensorID);
  if (layer < 0 || layer >= o2::itsmft::tracking::TrackerParamRef<DetId>::nLayers()) {
    throw std::runtime_error("Cluster layer is outside the detector");
  }

  const auto clusterData = o2::itsmft::ioutils::extractClusterData(cluster, patterns, dict);
  float sigma2Row = clusterData.sig2Row;
  float sigma2Col = clusterData.sig2Col;
  if (applySysErrors && shouldApplySysErrors<DetId>()) {
    addSysErrors<DetId>(layer, sigma2Row, sigma2Col);
  }

  if constexpr (DetId == o2::detectors::DetID::ITS) {
    const auto trkXYZ = geom->getMatrixT2L(sensorID) ^ clusterData.coordinates;
    const auto gloXYZ = geom->getMatrixL2G(sensorID) * clusterData.coordinates;
    result = {{gloXYZ.x(), gloXYZ.y(), gloXYZ.z()},
              {trkXYZ.x(), trkXYZ.y(), trkXYZ.z(), geom->getSensorRefAlpha(sensorID)},
              {sigma2Row, 0.f, sigma2Col},
              clusterData.nPixels,
              layer};
  } else {
    if (!geom->getCacheL2G().isFilled() || geom->getCacheL2G().getSize() <= sensorID) {
      throw std::runtime_error("Cluster geometry is not available");
    }
    const auto gloXYZ = geom->getMatrixL2G(sensorID) * clusterData.coordinates;
    result = {{gloXYZ.x(), gloXYZ.y(), gloXYZ.z()}, {}, {sigma2Row, 0.f, sigma2Col}, clusterData.nPixels, layer};
  }
  return result;
}

template <o2::detectors::DetID::ID DetId, typename Consume>
void decodeDetectorSource(const o2::itsmft::tracking::ClusterSourceInput& source, const Consume& consume)
{
  using Geometry = std::conditional_t<DetId == o2::detectors::DetID::ITS, o2::its::GeometryTGeo, o2::mft::GeometryTGeo>;
  Geometry* geometry = nullptr;
  if (!source.clusters.empty()) {
    geometry = Geometry::Instance();
    geometry->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));
  }
  consume([&](const auto& cluster, auto& patterns) {
    return decodeCluster<DetId>(geometry, cluster, patterns, source.dictionary, source.applySysErrors);
  });
}

} // namespace

namespace o2::itsmft::tracking
{
namespace
{
// Project decoded ITS facts into the accepted cylindrical convention.
GlobalMeasurement makeCylinderGlobalMeasurement(const DecodedCluster& decoded, uint32_t clusterId)
{
  const float sine = std::sin(decoded.cylinderFrame.frameAngle);
  const float cosine = std::cos(decoded.cylinderFrame.frameAngle);
  const auto& covariance = decoded.rowColumnCovariance;
  return GlobalMeasurement{
    decoded.global.x,
    decoded.global.y,
    decoded.global.z,
    {sine * sine * covariance.uu,
     -sine * cosine * covariance.uu,
     -sine * covariance.uv,
     cosine * cosine * covariance.uu,
     cosine * covariance.uv,
     covariance.vv},
    std::hypot(decoded.global.x, decoded.global.y),
    std::atan2(decoded.global.y, decoded.global.x),
    clusterId};
}

// Project decoded MFT facts into z-normal, global-x/global-y disk coordinates.
// ALPIDE row is established as global x and column as global y by the MFT
// geometry decoder. No legacy TrackingFrameInfo participates in this mapping.
GlobalMeasurement makeDiskGlobalMeasurement(const DecodedCluster& decoded, uint32_t clusterId)
{
  return GlobalMeasurement{
    decoded.global.x,
    decoded.global.y,
    decoded.global.z,
    {decoded.rowColumnCovariance.uu, decoded.rowColumnCovariance.uv, 0.f,
     decoded.rowColumnCovariance.vv, 0.f, 0.f},
    std::hypot(decoded.global.x, decoded.global.y),
    std::atan2(decoded.global.y, decoded.global.x),
    clusterId};
}

SurfaceMeasurement makeCylinderSurfaceMeasurement(const DecodedCluster& decoded)
{
  return {decoded.cylinderFrame, decoded.rowColumnCovariance};
}

SurfaceMeasurement makeDiskSurfaceMeasurement(const DecodedCluster& decoded)
{
  return {{decoded.global.z, decoded.global.x, decoded.global.y, 0.f},
          decoded.rowColumnCovariance};
}

bool covariance2DIsPositiveSemidefinite(float cxx, float cxy, float cyy) noexcept
{
  if (cxx < 0.f || cyy < 0.f) {
    return false;
  }
  const double diagonalProduct = static_cast<double>(cxx) * cyy;
  const double cxySquared = static_cast<double>(cxy) * cxy;
  const double tolerance = 16. * std::numeric_limits<float>::epsilon() *
                           std::max(diagonalProduct, cxySquared);
  return diagonalProduct - cxySquared >= -tolerance;
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
  return globalCovarianceIsPositiveSemidefinite(global.covariance) &
         covariance2DIsPositiveSemidefinite(local.covariance.uu, local.covariance.uv, local.covariance.vv);
}

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

} // namespace

namespace detail
{
void prepareSources(TimeFrame& frame, const SurfaceCatalogView& catalog,
                    gsl::span<const ClusterSourceInput> sources,
                    std::vector<std::vector<uint32_t>>* externalIndicesBySurface,
                    std::vector<std::vector<uint32_t>>* clusterSizesBySurface, bool requireCompleteMapping)
{
  clearFrameAndSidecars(frame, externalIndicesBySurface, clusterSizesBySurface);
  if (!frame.isConfigured()) {
    throw std::runtime_error("TimeFrame is not configured");
  }
  if (requireCompleteMapping && sources.empty()) {
    throw std::runtime_error("Malformed cluster loading input");
  }
  const auto nSources = static_cast<uint32_t>(sources.size());

  std::vector<bool> seen(nSources, false);
  std::vector<ClusterSourceId> sourceBySurface(catalog.nSurfaces, ClusterSourceId::invalid());
  for (const auto& src : sources) {
    if (!src.id.isValid() || src.id.value() >= nSources) {
      throw std::runtime_error(std::format("Source IDs must be dense source={}", src.id.value()));
    }
    if (seen[src.id.value()]) {
      throw std::runtime_error(std::format("Duplicate source ID source={}", src.id.value()));
    }
    seen[src.id.value()] = true;
    if (src.detector != o2::detectors::DetID::ITS && src.detector != o2::detectors::DetID::MFT) {
      throw std::runtime_error(std::format("Unsupported source detector source={}", src.id.value()));
    }
    if (!src.clusters.empty() && src.dictionary == nullptr) {
      throw std::runtime_error(std::format("Cluster dictionary is not available source={} rof={} clusterIndex={}", src.id.value(), 0, 0));
    }
    for (const auto surface : src.layerToSurface) {
      if (!surface.isValid() || surface.value() >= catalog.nSurfaces || surface.value() >= frame.getDetectorConfiguration().size()) {
        throw std::runtime_error(std::format("Invalid source-to-surface layer mapping source={}", src.id.value()));
      }
      if (sourceBySurface[surface.value()].isValid()) {
        throw std::runtime_error(std::format("Invalid source-to-surface layer mapping source={}", src.id.value()));
      }
      if (catalog.getSurface(surface).detectorId != static_cast<uint8_t>(src.detector)) {
        throw std::runtime_error(std::format("Source detector does not match its surface source={}", src.id.value()));
      }
      sourceBySurface[surface.value()] = src.id;
    }
  }
  if (requireCompleteMapping) {
    for (uint16_t position = 0; position < frame.getDetectorConfiguration().size(); ++position) {
      if (position < sourceBySurface.size() && sourceBySurface[position].isValid()) {
        continue;
      }
      // Attribute an omitted surface only when one source owns its detector.
      ClusterSourceId owner;
      for (const auto& source : sources) {
        if (static_cast<uint8_t>(source.detector) != frame.getDetectorConfiguration().getSurfaceCatalog().getSurface(LayerId{position}).detectorId) {
          continue;
        }
        if (owner.isValid()) {
          throw std::runtime_error("Invalid source-to-surface layer mapping");
        }
        owner = source.id;
      }
      throw std::runtime_error(std::format("Invalid source-to-surface layer mapping source={}", owner.value()));
    }
  }
  if (!sources.empty()) {
    frame.setROFViews(sources.front().rofViews);
  }
}
void validateSource(const ClusterSourceInput& src, const o2::InteractionRecord& origin)
{
  int64_t expectedNext = 0;
  for (uint32_t r = 0; r < src.rofs.size(); ++r) {
    const auto& rof = src.rofs[r];
    const int64_t first = rof.getFirstEntry();
    const int64_t n = rof.getNEntries();
    if (n < 0 || first != expectedNext) {
      throw std::runtime_error(std::format("Invalid ROF cluster range source={} rof={}", src.id.value(), r));
    }
    expectedNext = first + n;
    if (expectedNext > static_cast<int64_t>(src.clusters.size())) {
      throw std::runtime_error(std::format("Invalid ROF cluster range source={} rof={}", src.id.value(), r));
    }
  }
  if (expectedNext != static_cast<int64_t>(src.clusters.size())) {
    throw std::runtime_error(std::format("Invalid ROF cluster range source={} rof={}", src.id.value(), static_cast<uint32_t>(src.rofs.size())));
  }

  for (uint32_t r = 0; r < src.rofs.size(); ++r) {
    const auto built = computeROFIntervalBC(src.rofs[r].getBCData(), origin, src.timing, r);
    if (!built.ok()) {
      throw std::runtime_error(std::format("Invalid ROF timing: source={} rof={} timingError={}", src.id.value(), r, static_cast<int>(built.error)));
    }
  }
}
void appendCluster(TimeFrame& frame, const SurfaceCatalogView& catalog,
                   const ClusterSourceInput& src, const DecodedCluster& decoded,
                   uint32_t r, uint32_t externalIndex,
                   std::vector<std::vector<uint32_t>>& externalIndices,
                   std::vector<std::vector<uint32_t>>& clusterSizes)
{
  if (decoded.layer < 0 || static_cast<size_t>(decoded.layer) >= src.layerToSurface.size()) {
    throw std::runtime_error(std::format("Invalid source-to-surface layer mapping source={} rof={} clusterIndex={}", src.id.value(), r, externalIndex));
  }
  const auto expectedSurface = src.layerToSurface[decoded.layer];
  const auto& surfaceDescriptor = catalog.getSurface(expectedSurface);
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
    throw std::runtime_error(std::format("Malformed cluster loading input source={} rof={} clusterIndex={}", src.id.value(), r, externalIndex));
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
  clusterSizes[expectedSurface.value()].push_back(decoded.nPixels);
  externalIndices[expectedSurface.value()].push_back(externalIndex);
}
void bindSourceROFNavigation(TimeFrame& frame, const ClusterSourceInput& source,
                             const std::vector<std::vector<int>>& boundaries)
{
  for (uint16_t layer = 0; layer < source.layerToSurface.size(); ++layer) {
    frame.setROFNavigation(source.layerToSurface[layer].value(), boundaries[layer], source.rofViews, layer);
  }
}
} // namespace detail

void loadTimeFrameSources(TimeFrame& frame, gsl::span<const ClusterSourceInput> sources,
                          SurfaceCatalogView catalog, const o2::InteractionRecord& origin,
                          std::vector<std::vector<uint32_t>>* externalIndicesBySurface,
                          std::vector<std::vector<uint32_t>>* clusterSizesBySurface)
{
  detail::prepareSources(frame, catalog, sources, externalIndicesBySurface, clusterSizesBySurface, true);
  std::vector<std::vector<uint32_t>> externalIndices(catalog.nSurfaces);
  std::vector<std::vector<uint32_t>> clusterSizes(catalog.nSurfaces);
  bool hasMCInformation = false;
  for (const auto& source : sources) {
    detail::validateSource(source, origin);
    const auto load = [&](const auto& decode) {
      detail::loadDecodedSource(frame, catalog, source, decode, externalIndices, clusterSizes);
    };
    if (source.detector == o2::detectors::DetID::ITS) {
      decodeDetectorSource<o2::detectors::DetID::ITS>(source, load);
    } else {
      decodeDetectorSource<o2::detectors::DetID::MFT>(source, load);
    }
    hasMCInformation |= source.labels != nullptr;
  }
  frame.setHasMCInformation(hasMCInformation);
  if (externalIndicesBySurface != nullptr) {
    *externalIndicesBySurface = std::move(externalIndices);
  }
  if (clusterSizesBySurface != nullptr) {
    *clusterSizesBySurface = std::move(clusterSizes);
  }
}

} // namespace o2::itsmft::tracking
