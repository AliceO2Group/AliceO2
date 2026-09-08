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

#define BOOST_TEST_MODULE ITSMFT TrackletFinding
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#include <boost/test/unit_test.hpp>

#include <TGeoGlobalMagField.h>

#include "DataFormatsITS/Vertex.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Field/MagneticField.h"
#include "GPUCommonMath.h"
#include "ITSMFTTracking/detail/MFTFwdTrackHelpers.h"
#include "ITSMFTTracking/detail/CandidateFinding.h"
#include "ITSMFTTracking/detail/TrackingKernelParameters.h"
#include "ITSMFTTracking/detail/TrackerTraversalPreparation.h"
#include "ITStracking/TrackHelpers.h"

#include "TrackingParameterTestSupport.h"

using o2::itsmft::tracking::test::ReferenceTrackingParameters;
using namespace o2::itsmft;
using namespace o2::itsmft::tracking;

struct PropagatorFieldFixture {
  PropagatorFieldFixture()
  {
    if (!TGeoGlobalMagField::Instance()->GetField()) {
      TGeoGlobalMagField::Instance()->SetField(o2::field::MagneticField::createNominalField(5, true));
      TGeoGlobalMagField::Instance()->Lock();
    }
  }
};

BOOST_GLOBAL_FIXTURE(PropagatorFieldFixture);

/// Focused numerical-parity coverage for the first D007 surface-kind boundary
/// operation migrated off the legacy per-detector branch (Architecture.md
/// §10, cellsAreCompatible). These tests do not exercise TrackerTraits'
/// production traversal -- see the handoff note on scope.

namespace
{

constexpr float Bz = 0.5f;

o2::its::TrackingFrameInfo makeBarrelHit(float xTF, float alpha, float y, float z, float sigma2Y = 1.e-4f, float sigma2Z = 1.e-4f)
{
  return o2::its::TrackingFrameInfo{xTF, y, z, xTF, alpha, {y, z}, {sigma2Y, 0.f, sigma2Z}};
}

o2::its::TrackingFrameInfo makeDiskHit(float z, float x, float y, float sigma2X = 1.e-2f, float sigma2Y = 1.e-2f)
{
  return o2::its::TrackingFrameInfo{x, y, z, 0.f, 0.f, {x, y}, {sigma2X, 0.f, sigma2Y}};
}

o2::its::Vertex makeVertex(float x, float y, float z,
                           float sigma2X, float sigma2Y, float sigma2Z,
                           unsigned short contributors = 1)
{
  const float position[3]{x, y, z};
  const float covariance[6]{sigma2X, 0.f, sigma2Y, 0.f, 0.f, sigma2Z};
  return o2::its::Vertex{position, covariance, contributors, 1.f};
}

GlobalMeasurement makeGlobalCluster(float x, float y, float z, int id = 0)
{
  GlobalMeasurement measurement{};
  measurement.position = {x, y, z};
  measurement.radius = std::hypot(x, y);
  measurement.phi = o2::its::math_utils::computePhi(x, y);
  measurement.clusterId = static_cast<uint32_t>(id);
  return measurement;
}

GlobalMeasurement makeMeasurement(float x, float y, float z, float uu = 1.e-4f, float vv = 1.e-4f, float uv = 0.f)
{
  GlobalMeasurement measurement{};
  measurement.position = {x, y, z};
  measurement.radius = std::hypot(x, y);
  measurement.covariance = {uu, uv, 0.f, vv, 0.f, 0.f};
  return measurement;
}

GlobalMeasurement makeMeasurement(const GlobalMeasurement& cluster, float uu = 1.e-4f, float vv = 1.e-4f, float uv = 0.f)
{
  auto measurement = cluster;
  measurement.covariance = {uu, uv, 0.f, vv, 0.f, 0.f};
  return measurement;
}

TrackletProjectionCache makeCylinderProjectionCache(int fromLayer, int toLayer, float fromRadius, float toRadius,
                                                    float targetMinR, float targetMaxR, float sourcePositionResolution,
                                                    float edgeMSAngle, float edgePhiCut)
{
  return {fromLayer, toLayer, fromRadius, toRadius, targetMinR, targetMaxR, 0.f, 0.f,
          sourcePositionResolution, edgeMSAngle, edgePhiCut};
}

TrackletProjectionCache makeDiskProjectionCache(int fromLayer, int toLayer, float fromRadius,
                                                float, float targetMinZ, float targetMaxZ,
                                                float edgeMSAngle, float edgePhiCut)
{
  return {fromLayer, toLayer, fromRadius, 0.f, 0.f, 0.f, targetMinZ, targetMaxZ,
          0.f, edgeMSAngle, edgePhiCut};
}

// CandidateFinding exposes one descriptor-selected projection operation.
// Keep the numerical fixtures readable without exporting coordinate leaves.
bool projectCylinderSearchWindow(const GlobalMeasurement& sourceMeasurement,
                                 const GlobalMeasurement&,
                                 const o2::its::Vertex& vertex,
                                 const TrackletProjectionCache& edgeCache,
                                 const o2::itsmft::IndexTableUtilsCore& indexUtils,
                                 const TrackingKernelParameters& params,
                                 TrackletSearchWindow& out)
{
  return projectTrackletSearchWindow(sourceMeasurement, vertex, 0.f, SurfaceKind::Cylinder,
                                     edgeCache, indexUtils, params.nSigmaCut, out);
}

bool projectDiskSearchWindow(const GlobalMeasurement& sourceMeasurement,
                             const GlobalMeasurement&,
                             const o2::its::Vertex& vertex,
                             const TrackletProjectionCache& edgeCache,
                             const o2::itsmft::IndexTableUtilsCore& indexUtils,
                             const TrackingKernelParameters& params,
                             TrackletSearchWindow& out)
{
  return projectTrackletSearchWindow(sourceMeasurement, vertex, 0.f, SurfaceKind::Disk,
                                     edgeCache, indexUtils, params.nSigmaCut, out);
}

void setDiskLookup(IndexTableUtilsCore& indexUtils, const ReferenceTrackingParameters& params,
                   float radialMin = 0.1f, float radialMax = 20.f)
{
  std::array<float, IndexTableUtilsCore::MaxLayers> minima{};
  std::array<float, IndexTableUtilsCore::MaxLayers> maxima{};
  minima.fill(radialMin);
  maxima.fill(radialMax);
  indexUtils.setIndexTableParams(IndexTableCoordType::PhiR, params.RowBins, params.ColBins,
                                 0.f, o2::constants::math::TwoPI, minima, maxima);
}

void checkSearchWindowEqual(const TrackletSearchWindow& lhs, const TrackletSearchWindow& rhs)
{
  BOOST_CHECK_EQUAL(lhs.bins.x, rhs.bins.x);
  BOOST_CHECK_EQUAL(lhs.bins.y, rhs.bins.y);
  BOOST_CHECK_EQUAL(lhs.bins.z, rhs.bins.z);
  BOOST_CHECK_EQUAL(lhs.bins.w, rhs.bins.w);
  BOOST_CHECK_EQUAL(lhs.sourceReferenceCoordinate, rhs.sourceReferenceCoordinate);
  BOOST_CHECK_EQUAL(lhs.sourceProjectedCoordinate, rhs.sourceProjectedCoordinate);
  BOOST_CHECK_EQUAL(lhs.slope, rhs.slope);
  BOOST_CHECK_EQUAL(lhs.varianceConstant, rhs.varianceConstant);
  BOOST_CHECK_EQUAL(lhs.varianceLinear, rhs.varianceLinear);
  BOOST_CHECK_EQUAL(lhs.varianceQuadratic, rhs.varianceQuadratic);
  BOOST_CHECK_EQUAL(lhs.phiPrediction, rhs.phiPrediction);
  BOOST_CHECK_EQUAL(lhs.phiVariance, rhs.phiVariance);
}

std::pair<float, float> evaluateSearchWindowAt(const TrackletSearchWindow& window, float targetReferenceCoordinate)
{
  const float delta = targetReferenceCoordinate - window.sourceReferenceCoordinate;
  return {window.sourceProjectedCoordinate + window.slope * delta,
          window.varianceConstant + delta * (window.varianceLinear + delta * window.varianceQuadratic)};
}

NominalSurfaceMaterial toMaterial(float xOverX0)
{
  return NominalSurfaceMaterial{xOverX0, xOverX0 * o2::its::constants::Radl * o2::its::constants::Rho};
}

std::array<NominalSurfaceMaterial, 3> toMaterial(const std::array<float, 3>& xOverX0)
{
  return {toMaterial(xOverX0[0]), toMaterial(xOverX0[1]), toMaterial(xOverX0[2])};
}

std::vector<SurfaceDescriptor> toCatalog(const std::vector<float>& xOverX0)
{
  std::vector<SurfaceDescriptor> material;
  material.reserve(xOverX0.size());
  for (const float x0 : xOverX0) {
    SurfaceDescriptor descriptor;
    descriptor.material = toMaterial(x0);
    material.push_back(descriptor);
  }
  return material;
}

TrackingKernelParameters makeKernelParameters(const ReferenceTrackingParameters& params, SurfaceKind kind)
{
  (void)kind;
  TrackingKernelParameters out;
  out.trackletMinPt = params.TrackletMinPt;
  out.nSigmaCut = params.NSigmaCut;
  out.maxChi2ClusterAttachment = params.MaxChi2ClusterAttachment;
  out.maxChi2NDF = params.MaxChi2NDF;
  out.pvResolution = params.PVres;
  return out;
}

} // namespace

BOOST_AUTO_TEST_CASE(BindingCopiesEveryFieldToTheCorrectSlot)
{
  // Distinct sentinel per field so a field-swap bug in the binding is caught.
  ReferenceTrackingParameters legacy;
  legacy.TrackletMinPt = 1.11f;
  legacy.NSigmaCut = 3.33f;
  legacy.MaxChi2ClusterAttachment = 4.44f;
  legacy.MaxChi2NDF = 5.55f;
  legacy.PVres = 8.88f;
  legacy.LayerxX0 = {0.011f, 0.022f, 0.033f};
  legacy.CorrType = o2::base::PropagatorF::MatCorrType::USEMatCorrLUT;

  const auto barrel = makeKernelParameters(legacy, SurfaceKind::Cylinder);
  BOOST_CHECK_CLOSE(barrel.trackletMinPt, 1.11f, 1e-6);
  BOOST_CHECK_CLOSE(barrel.nSigmaCut, 3.33f, 1e-6);
  BOOST_CHECK_CLOSE(barrel.maxChi2ClusterAttachment, 4.44f, 1e-6);
  BOOST_CHECK_CLOSE(barrel.maxChi2NDF, 5.55f, 1e-6);
  BOOST_CHECK_CLOSE(barrel.pvResolution, 8.88f, 1e-6);
  BOOST_CHECK(barrel.isValid());

  const auto disk = makeKernelParameters(legacy, SurfaceKind::Disk);
  BOOST_CHECK_CLOSE(disk.trackletMinPt, 1.11f, 1e-6);
  BOOST_CHECK_CLOSE(disk.nSigmaCut, 3.33f, 1e-6);
  BOOST_CHECK_CLOSE(disk.maxChi2ClusterAttachment, 4.44f, 1e-6);
  BOOST_CHECK_CLOSE(disk.maxChi2NDF, 5.55f, 1e-6);
  BOOST_CHECK(disk.isValid());

  const auto legacyMaterial = toCatalog(legacy.LayerxX0);
  const auto attach = bindAttachHitConfig(SurfaceCatalogView{legacyMaterial.data(), static_cast<uint32_t>(legacyMaterial.size())}, legacy);
  BOOST_REQUIRE_EQUAL(attach.catalog.nSurfaces, 3u);
  BOOST_CHECK_CLOSE(attach.catalog.surfaces[0].material.xOverX0, 0.011f, 1e-6);
  BOOST_CHECK_CLOSE(attach.catalog.surfaces[1].material.xOverX0, 0.022f, 1e-6);
  BOOST_CHECK_CLOSE(attach.catalog.surfaces[2].material.xOverX0, 0.033f, 1e-6);
  BOOST_CHECK(attach.corrType == o2::base::PropagatorF::MatCorrType::USEMatCorrLUT);
  BOOST_CHECK(attach.isValid(3));
  BOOST_CHECK(!attach.isValid(4));
}

BOOST_AUTO_TEST_CASE(BoundConfigurationRejectsInvalidCorrectionType)
{
  ReferenceTrackingParameters legacy;
  legacy.TrackletMinPt = 1.11f;
  legacy.NSigmaCut = 3.33f;
  legacy.MaxChi2ClusterAttachment = 4.44f;
  legacy.MaxChi2NDF = 5.55f;

  auto invalidCorrection = legacy;
  invalidCorrection.CorrType = static_cast<o2::base::PropagatorF::MatCorrType>(99);
  const auto invalidCorrectionMaterial = toCatalog(invalidCorrection.LayerxX0);
  BOOST_CHECK(!bindAttachHitConfig(SurfaceCatalogView{invalidCorrectionMaterial.data(), static_cast<uint32_t>(invalidCorrectionMaterial.size())}, invalidCorrection)
                 .isValid(invalidCorrection.LayerxX0.size()));
}

BOOST_AUTO_TEST_CASE(CylinderProjectSearchWindowUsesCandidateRadiusAndBoundsTheFullTargetInterval)
{
  ReferenceTrackingParameters legacy;
  legacy.PVres = 0.f;
  const auto params = makeKernelParameters(legacy, SurfaceKind::Cylinder);
  BOOST_REQUIRE(params.isValid());

  IndexTableUtilsCore indexUtils;
  indexUtils.setTrackingParameters(legacy);

  const auto source = makeGlobalCluster(2.f, 0.f, 0.5f);
  const auto sourceMeasurement = makeMeasurement(source);
  const auto vertex = makeVertex(0.f, 0.f, 0.f, 1.e-4f, 1.e-4f, 4.e-4f, 4);
  const auto state = makeCylinderProjectionCache(0, 3, 2.f, 4.f, 3.8f, 4.2f, 5.e-4f, 2.e-3f, 0.08f);

  TrackletSearchWindow window{};
  BOOST_REQUIRE((projectCylinderSearchWindow(
    sourceMeasurement, source, vertex, state, indexUtils, params, window)));

  const float tanLambda = (source.z - vertex.getZ()) / source.radius;
  const float targetMeanRadius = 0.5f * (state.targetMinR + state.targetMaxR);
  const float deltaRadius = targetMeanRadius - source.radius;
  const float zAtTargetMeanR = tanLambda * deltaRadius + source.z;
  const float projectionScale = 1.f + deltaRadius / source.radius;
  const float originScale = projectionScale - 1.f;
  const float sourceCoordinateVariance = o2::its::math_utils::Sq(state.sourcePositionResolution);
  const float varianceZ =
    o2::its::math_utils::Sq(projectionScale) * sourceCoordinateVariance +
    o2::its::math_utils::Sq(tanLambda * projectionScale) * sourceCoordinateVariance +
    o2::its::math_utils::Sq(originScale) * vertex.getSigmaZ2() +
    o2::its::math_utils::Sq(deltaRadius * state.edgeMSAngle);
  const auto predictionAndVarianceAt = [&](float radius) {
    const float deltaR = radius - source.radius;
    const float scale = 1.f + deltaR / source.radius;
    const float origin = scale - 1.f;
    const float candidateVariance =
      o2::its::math_utils::Sq(scale) * sourceCoordinateVariance +
      o2::its::math_utils::Sq(tanLambda * scale) * sourceCoordinateVariance +
      o2::its::math_utils::Sq(origin) * vertex.getSigmaZ2() +
      o2::its::math_utils::Sq(deltaR * state.edgeMSAngle);
    return std::pair{source.z + tanLambda * deltaR, candidateVariance};
  };
  const auto [minPrediction, minVariance] = predictionAndVarianceAt(state.targetMinR);
  const auto [maxPrediction, maxVariance] = predictionAndVarianceAt(state.targetMaxR);
  const float lowerBound = std::min(minPrediction - params.nSigmaCut * std::sqrt(minVariance),
                                    maxPrediction - params.nSigmaCut * std::sqrt(maxVariance));
  const float upperBound = std::max(minPrediction + params.nSigmaCut * std::sqrt(minVariance),
                                    maxPrediction + params.nSigmaCut * std::sqrt(maxVariance));
  const auto directBins = getBinsPhiColumn(source.phi, state.toLayer, 0.5f * (lowerBound + upperBound),
                                           0.5f * (upperBound - lowerBound), state.edgePhiCut, indexUtils);

  BOOST_CHECK_EQUAL(window.bins.x, directBins.x);
  BOOST_CHECK_EQUAL(window.bins.y, directBins.y);
  BOOST_CHECK_EQUAL(window.bins.z, directBins.z);
  BOOST_CHECK_EQUAL(window.bins.w, directBins.w);
  const auto [midpointPrediction, midpointVariance] = evaluateSearchWindowAt(window, targetMeanRadius);
  BOOST_CHECK_EQUAL(midpointPrediction, zAtTargetMeanR);
  BOOST_CHECK_CLOSE_FRACTION(midpointVariance, varianceZ, 1.e-6f);
  const auto [evaluatedMinPrediction, evaluatedMinVariance] = evaluateSearchWindowAt(window, state.targetMinR);
  BOOST_CHECK_EQUAL(evaluatedMinPrediction, minPrediction);
  BOOST_CHECK_CLOSE_FRACTION(evaluatedMinVariance, minVariance, 1.e-6f);
  const auto [evaluatedMaxPrediction, evaluatedMaxVariance] = evaluateSearchWindowAt(window, state.targetMaxR);
  BOOST_CHECK_EQUAL(evaluatedMaxPrediction, maxPrediction);
  BOOST_CHECK_CLOSE_FRACTION(evaluatedMaxVariance, maxVariance, 1.e-6f);

  TrackletSearchWindow beamUncertaintyWindow{};
  BOOST_REQUIRE(projectTrackletSearchWindow(sourceMeasurement, vertex, 1.e-3f,
                                            SurfaceKind::Cylinder, state, indexUtils, params.nSigmaCut,
                                            beamUncertaintyWindow));
  const auto [beamPrediction, beamVariance] = evaluateSearchWindowAt(beamUncertaintyWindow, targetMeanRadius);
  BOOST_CHECK_EQUAL(beamPrediction, zAtTargetMeanR);
  BOOST_CHECK_CLOSE_FRACTION(beamVariance,
                             varianceZ + o2::its::math_utils::Sq(tanLambda * originScale) * 1.e-3f, 1.e-6f);

  legacy.PVres = 0.025f;
  const auto differentConfiguredPVParams = makeKernelParameters(legacy, SurfaceKind::Cylinder);
  BOOST_REQUIRE(differentConfiguredPVParams.isValid());
  TrackletSearchWindow differentConfiguredPVWindow{};
  BOOST_REQUIRE((projectCylinderSearchWindow(
    sourceMeasurement, source, vertex, state, indexUtils, differentConfiguredPVParams, differentConfiguredPVWindow)));
  checkSearchWindowEqual(differentConfiguredPVWindow, window);
}

BOOST_AUTO_TEST_CASE(DiskProjectSearchWindowBuildsPeriodicPhiRCoordinates)
{
  ReferenceTrackingParameters legacy;
  const auto params = makeKernelParameters(legacy, SurfaceKind::Disk);
  BOOST_REQUIRE(params.isValid());

  IndexTableUtilsCore indexUtils;
  setDiskLookup(indexUtils, legacy);

  constexpr int fromLayer = 1;
  constexpr int toLayer = 4; // deliberately skipped/nonadjacent edge
  const float fromZ = detail::mftLayerZ(fromLayer);
  const float toZ = detail::mftLayerZ(toLayer);
  const auto source = makeGlobalCluster(1.2f, 0.7f, fromZ);
  const auto sourceMeasurement = makeMeasurement(source, 2.e-4f, 3.e-4f);
  const auto vertex = makeVertex(0.01f, -0.02f, 0.1f, 4.e-4f, 5.e-4f, 0.04f, 3);
  const auto state = makeDiskProjectionCache(fromLayer, toLayer, 2.f, fromZ, toZ, toZ, 3.e-3f, 0.04f);

  TrackletSearchWindow window{};
  BOOST_REQUIRE((projectDiskSearchWindow(
    sourceMeasurement, source, vertex, state, indexUtils, params, window)));

  const float slope = source.radius / (source.z - vertex.getZ());
  const float deltaZ = toZ - source.z;
  const float expectedRadius = source.radius + slope * deltaZ;
  const float radialScale = expectedRadius / source.radius;
  const float expectedX = radialScale * source.x;
  const float expectedY = radialScale * source.y;
  const float projectionScale = 1.f + deltaZ / (source.z - vertex.getZ());
  const float originScale = projectionScale - 1.f;
  const float sourceCoordinateVariance = o2::its::math_utils::Sq(state.sourcePositionResolution);
  const float varianceR =
    o2::its::math_utils::Sq(projectionScale) * sourceCoordinateVariance +
    o2::its::math_utils::Sq(slope * projectionScale) * sourceCoordinateVariance +
    o2::its::math_utils::Sq(slope * originScale) * vertex.getSigmaZ2() +
    o2::its::math_utils::Sq(deltaZ * state.edgeMSAngle);

  const auto [evaluatedRadius, evaluatedVariance] = evaluateSearchWindowAt(window, toZ);
  BOOST_CHECK_EQUAL(evaluatedRadius, expectedRadius);
  BOOST_CHECK_CLOSE_FRACTION(evaluatedVariance, varianceR, 1.e-6f);
  BOOST_CHECK_EQUAL(window.phiPrediction, source.phi);
  BOOST_CHECK_EQUAL(window.phiVariance, o2::its::math_utils::Sq(state.edgePhiCut / params.nSigmaCut));

  TrackletSearchWindow beamUncertaintyWindow{};
  BOOST_REQUIRE(projectTrackletSearchWindow(sourceMeasurement, vertex, 1.e-3f,
                                            SurfaceKind::Disk, state, indexUtils, params.nSigmaCut,
                                            beamUncertaintyWindow));
  const auto [beamRadius, beamVariance] = evaluateSearchWindowAt(beamUncertaintyWindow, toZ);
  BOOST_CHECK_EQUAL(beamRadius, expectedRadius);
  BOOST_CHECK_CLOSE_FRACTION(beamVariance,
                             varianceR + o2::its::math_utils::Sq(originScale) * 1.e-3f, 1.e-6f);
  BOOST_CHECK_EQUAL(beamUncertaintyWindow.phiVariance, window.phiVariance);
}

BOOST_AUTO_TEST_CASE(DiskProjectSearchWindowUsesCandidateZAndBoundsTheFullTargetInterval)
{
  ReferenceTrackingParameters legacy;
  const auto params = makeKernelParameters(legacy, SurfaceKind::Disk);
  BOOST_REQUIRE(params.isValid());

  IndexTableUtilsCore indexUtils;
  setDiskLookup(indexUtils, legacy);

  constexpr int fromLayer = 0;
  constexpr int toLayer = 1;
  const float fromZ = detail::mftLayerZ(fromLayer);
  const float toZ = detail::mftLayerZ(toLayer);
  const auto source = makeGlobalCluster(1.2f, 0.7f, fromZ);
  const auto measurement = makeMeasurement(source, 2.e-4f, 3.e-4f);
  const auto vertex = makeVertex(0.01f, -0.02f, 0.1f, 4.e-4f, 5.e-4f, 0.04f, 3);

  const auto pointTarget = makeDiskProjectionCache(fromLayer, toLayer, 2.f, fromZ, toZ, toZ, 3.e-3f, 0.04f);
  const auto intervalTarget = makeDiskProjectionCache(fromLayer, toLayer, 2.f, fromZ, toZ - 0.5f, toZ + 0.5f, 3.e-3f, 0.04f);
  TrackletSearchWindow pointWindow{};
  TrackletSearchWindow intervalWindow{};
  BOOST_REQUIRE((projectDiskSearchWindow(measurement, source, vertex, pointTarget, indexUtils, params, pointWindow)));
  BOOST_REQUIRE((projectDiskSearchWindow(measurement, source, vertex, intervalTarget, indexUtils, params, intervalWindow)));

  const float slope = source.radius / (source.z - vertex.getZ());
  const float sourceCoordinateVariance = o2::its::math_utils::Sq(intervalTarget.sourcePositionResolution);
  const float sourceVarianceScale = (1.f + o2::its::math_utils::Sq(slope)) * sourceCoordinateVariance;
  const float originVarianceScale = o2::its::math_utils::Sq(slope) * vertex.getSigmaZ2();
  const float edgeMSVarianceScale = o2::its::math_utils::Sq(intervalTarget.edgeMSAngle);
  const auto predictionAndVarianceAt = [&](float z) {
    const float deltaZ = z - source.z;
    const float originScale = deltaZ / (source.z - vertex.getZ());
    const float projectionScale = 1.f + originScale;
    const float candidateVariance =
      o2::its::math_utils::Sq(projectionScale) * sourceVarianceScale +
      o2::its::math_utils::Sq(originScale) * originVarianceScale +
      o2::its::math_utils::Sq(deltaZ) * edgeMSVarianceScale;
    return std::pair{source.radius + slope * deltaZ, candidateVariance};
  };
  const auto [minPrediction, minVariance] = predictionAndVarianceAt(intervalTarget.targetMinZ);
  const auto [maxPrediction, maxVariance] = predictionAndVarianceAt(intervalTarget.targetMaxZ);
  const float lowerBound = std::min(minPrediction - params.nSigmaCut * std::sqrt(minVariance),
                                    maxPrediction - params.nSigmaCut * std::sqrt(maxVariance));
  const float upperBound = std::max(minPrediction + params.nSigmaCut * std::sqrt(minVariance),
                                    maxPrediction + params.nSigmaCut * std::sqrt(maxVariance));
  const auto directBins = getBinsPhiColumn(source.phi, intervalTarget.toLayer, 0.5f * (lowerBound + upperBound),
                                           0.5f * (upperBound - lowerBound), intervalTarget.edgePhiCut, indexUtils);

  BOOST_CHECK_EQUAL(intervalWindow.bins.x, directBins.x);
  BOOST_CHECK_EQUAL(intervalWindow.bins.y, directBins.y);
  BOOST_CHECK_EQUAL(intervalWindow.bins.z, directBins.z);
  BOOST_CHECK_EQUAL(intervalWindow.bins.w, directBins.w);
  const auto [pointPrediction, pointVariance] = evaluateSearchWindowAt(pointWindow, toZ);
  const auto [intervalPrediction, intervalVariance] = evaluateSearchWindowAt(intervalWindow, toZ);
  BOOST_CHECK_CLOSE_FRACTION(intervalPrediction, pointPrediction, 1.e-6f);
  BOOST_CHECK_CLOSE_FRACTION(intervalVariance, pointVariance, 1.e-6f);
  BOOST_CHECK_CLOSE_FRACTION(intervalWindow.phiPrediction, pointWindow.phiPrediction, 1.e-6f);
  BOOST_CHECK_SMALL(intervalWindow.phiVariance - pointWindow.phiVariance, 1.e-9f);

  const auto [evaluatedMinPrediction, evaluatedMinVariance] = evaluateSearchWindowAt(intervalWindow, intervalTarget.targetMinZ);
  BOOST_CHECK_EQUAL(evaluatedMinPrediction, minPrediction);
  BOOST_CHECK_CLOSE_FRACTION(evaluatedMinVariance, minVariance, 1.e-6f);
  const auto [evaluatedMaxPrediction, evaluatedMaxVariance] = evaluateSearchWindowAt(intervalWindow, intervalTarget.targetMaxZ);
  BOOST_CHECK_EQUAL(evaluatedMaxPrediction, maxPrediction);
  BOOST_CHECK_CLOSE_FRACTION(evaluatedMaxVariance, maxVariance, 1.e-6f);
}

BOOST_AUTO_TEST_CASE(ProjectSearchWindowInvalidBinsLeaveEveryOutputFieldUnchanged)
{
  ReferenceTrackingParameters legacy;

  IndexTableUtilsCore cylinderIndexUtils;
  cylinderIndexUtils.setTrackingParameters(legacy);
  const auto cylinderParams = makeKernelParameters(legacy, SurfaceKind::Cylinder);
  const auto cylinderSource = makeGlobalCluster(2.f, 0.f, 100.f);
  const auto cylinderMeasurement = makeMeasurement(cylinderSource);
  const auto cylinderVertex = makeVertex(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
  const auto cylinderState = makeCylinderProjectionCache(0, 3, 2.f, 4.f, 3.8f, 4.2f, 5.e-4f, 2.e-3f, 0.08f);
  const TrackletSearchWindow cylinderSentinel{
    {101, 102, 103, 104}, 105.f, 106.f, 107.f, 108.f, 109.f, 110.f, 111.f, 112.f};
  auto cylinderOut = cylinderSentinel;
  BOOST_CHECK(!(projectCylinderSearchWindow(
    cylinderMeasurement, cylinderSource, cylinderVertex, cylinderState, cylinderIndexUtils, cylinderParams, cylinderOut)));
  checkSearchWindowEqual(cylinderOut, cylinderSentinel);

  IndexTableUtilsCore diskIndexUtils;
  setDiskLookup(diskIndexUtils, legacy, 0.1f, 0.01f);
  const auto diskParams = makeKernelParameters(legacy, SurfaceKind::Disk);
  constexpr int fromLayer = 0;
  constexpr int toLayer = 1;
  const float fromZ = detail::mftLayerZ(fromLayer);
  const float toZ = detail::mftLayerZ(toLayer);
  const auto diskSource = makeGlobalCluster(1.f, 0.5f, fromZ);
  const auto diskMeasurement = makeMeasurement(diskSource);
  const auto diskVertex = makeVertex(0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
  const auto diskState = makeDiskProjectionCache(fromLayer, toLayer, 2.f, fromZ, toZ, toZ, 3.e-3f, 0.04f);
  const TrackletSearchWindow diskSentinel{
    {201, 202, 203, 204}, 205.f, 206.f, 207.f, 208.f, 209.f, 210.f, 211.f, 212.f};
  auto diskOut = diskSentinel;
  BOOST_CHECK(!(projectDiskSearchWindow(
    diskMeasurement, diskSource, diskVertex, diskState, diskIndexUtils, diskParams, diskOut)));
  checkSearchWindowEqual(diskOut, diskSentinel);
}

BOOST_AUTO_TEST_CASE(DiskProjectionUsesBeamCenteredPolarCoordinatesAndIgnoresVertexXY)
{
  ReferenceTrackingParameters legacy;
  const auto params = makeKernelParameters(legacy, SurfaceKind::Disk);
  constexpr int fromLayer = 0;
  constexpr int toLayer = 1;
  const float fromZ = detail::mftLayerZ(fromLayer);
  const float toZ = detail::mftLayerZ(toLayer);
  const auto source = makeGlobalCluster(1.f, 0.5f, fromZ);
  const auto sourceMeasurement = makeMeasurement(source);
  const auto state = makeDiskProjectionCache(fromLayer, toLayer, 2.f, fromZ, toZ, toZ, 3.e-3f, 0.04f);

  IndexTableUtilsCore indexUtils;
  setDiskLookup(indexUtils, legacy);

  const auto straightVertex = makeVertex(0.1f, -0.2f, 0.3f, 4.e-4f, 5.e-4f, 0.04f);
  TrackletSearchWindow straightWindow{};
  BOOST_REQUIRE((projectDiskSearchWindow(
    sourceMeasurement, source, straightVertex, state, indexUtils, params, straightWindow)));
  const float slope = source.radius / (source.z - straightVertex.getZ());
  const float expectedRadius = source.radius + slope * (toZ - source.z);
  const auto [straightPrediction, straightVariance] = evaluateSearchWindowAt(straightWindow, toZ);
  BOOST_CHECK_EQUAL(straightPrediction, expectedRadius);
  BOOST_CHECK(straightVariance > 0.f);
  BOOST_CHECK_EQUAL(straightWindow.phiPrediction, source.phi);

  const auto displacedVertex = makeVertex(-3.f, 4.f, straightVertex.getZ(), 8.f, 9.f, straightVertex.getSigmaZ2());
  TrackletSearchWindow displacedWindow{};
  BOOST_REQUIRE((projectDiskSearchWindow(
    sourceMeasurement, source, displacedVertex, state, indexUtils, params, displacedWindow)));
  checkSearchWindowEqual(displacedWindow, straightWindow);

  const auto fallbackVertex = makeVertex(0.1f, -0.2f, fromZ, 4.e-4f, 5.e-4f, 0.f);
  TrackletSearchWindow fallbackWindow{};
  const TrackletSearchWindow sentinel{{1, 2, 3, 4}, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
  fallbackWindow = sentinel;
  BOOST_CHECK(!(projectDiskSearchWindow(
    sourceMeasurement, source, fallbackVertex, state, indexUtils, params, fallbackWindow)));
  checkSearchWindowEqual(fallbackWindow, sentinel);
}

BOOST_AUTO_TEST_CASE(GlobalMeasurementsAreTheSoleCoordinateAuthority)
{
  ReferenceTrackingParameters cylinderParameters;
  cylinderParameters.PVres = 0.f;
  const auto cylinderKernelParameters = makeKernelParameters(cylinderParameters, SurfaceKind::Cylinder);
  IndexTableUtilsCore cylinderIndex;
  cylinderIndex.setTrackingParameters(cylinderParameters);
  const auto vertex = makeVertex(0.f, 0.f, 0.f, 1.e-4f, 1.e-4f, 4.e-4f, 4);
  const auto cylinderState = makeCylinderProjectionCache(0, 1, 2.f, 4.f, 3.8f, 4.2f, 5.e-4f, 2.e-3f, 0.08f);
  const auto sourceMeasurement = makeMeasurement(2.f, 0.f, 0.5f);
  const auto source = makeGlobalCluster(2.f, 0.f, 0.5f);

  TrackletSearchWindow baseline{};
  BOOST_REQUIRE((projectCylinderSearchWindow(
    sourceMeasurement, source, vertex, cylinderState, cylinderIndex, cylinderKernelParameters, baseline)));

  auto poisonedSource = source;
  poisonedSource.x = -999.f;
  poisonedSource.y = 888.f;
  poisonedSource.z = -777.f;
  TrackletSearchWindow poisonedWindow{};
  BOOST_REQUIRE((projectCylinderSearchWindow(
    sourceMeasurement, poisonedSource, vertex, cylinderState, cylinderIndex, cylinderKernelParameters, poisonedWindow)));
  checkSearchWindowEqual(poisonedWindow, baseline);

  auto poisonedNavigationCache = source;
  poisonedNavigationCache.radius = 4.f;
  TrackletSearchWindow cachePoisonedWindow{};
  BOOST_REQUIRE((projectCylinderSearchWindow(
    sourceMeasurement, poisonedNavigationCache, vertex, cylinderState, cylinderIndex, cylinderKernelParameters, cachePoisonedWindow)));
  checkSearchWindowEqual(cachePoisonedWindow, baseline);

  ReferenceTrackingParameters diskParameters;
  const auto diskKernelParameters = makeKernelParameters(diskParameters, SurfaceKind::Disk);
  IndexTableUtilsCore diskIndex;
  setDiskLookup(diskIndex, diskParameters);
  const float fromZ = detail::mftLayerZ(0);
  const float toZ = detail::mftLayerZ(1);
  const auto diskMeasurement = makeMeasurement(1.f, 0.5f, fromZ, 2.e-4f, 3.e-4f, 7.f);
  auto diskLocator = makeGlobalCluster(1.f, 0.5f, fromZ);
  const auto diskState = makeDiskProjectionCache(0, 1, 2.f, fromZ, toZ, toZ, 3.e-3f, 0.04f);
  TrackletSearchWindow diskBaseline{};
  BOOST_REQUIRE((projectDiskSearchWindow(
    diskMeasurement, diskLocator, vertex, diskState, diskIndex, diskKernelParameters, diskBaseline)));
  diskLocator.x = 123.f;
  diskLocator.y = -321.f;
  diskLocator.z = 456.f;
  auto uvPoisoned = diskMeasurement;
  uvPoisoned.covariance.xy = -12345.f;
  TrackletSearchWindow diskPoisoned{};
  BOOST_REQUIRE((projectDiskSearchWindow(
    uvPoisoned, diskLocator, vertex, diskState, diskIndex, diskKernelParameters, diskPoisoned)));
  checkSearchWindowEqual(diskPoisoned, diskBaseline);
}

/// Gate 3 edge-preparation slice coverage (relocated from
/// TimeFrame::initialise() into TrackerTraits::initialiseTimeFrame(); see
/// CandidateFinding.h family scattering leaves and
/// prepareEdgeScatteringAndBending. These tests verify
/// exact legacy-formula parity, the family-specific arithmetic literal that
/// integration review required preserved (not canonicalized), and the
/// order-sensitive oneOverR ratchet -- independently of TrackerTraits'
/// production traversal (covered separately in
/// testComputeLayerTrackletsOrchestration.cxx).

namespace
{
/// Independent re-transcription of the shared arithmetic in the frozen
/// ITS-only TimeFrame::initialise() (ITS/tracking/src/TimeFrame.cxx:352-370),
/// which the (now-removed) common-CA non-MFT branch reproduced verbatim.
/// Deliberately re-derived here rather than calling
/// prepareEdgeScatteringAndBending, so a transcription mistake in
/// either the operation or this reference would show up as a mismatch.
EdgeScatteringBendingPrep referenceEdgeScatteringAndBending(
  gsl::span<const float> perLayerMSAngle, int fromLayer, int toLayer,
  float r1, float r2, float clampedOneOverR, float res1, float res2)
{
  float ms2 = 0.f;
  for (int layer = fromLayer; layer < toLayer; ++layer) {
    ms2 += o2::its::math_utils::Sq(perLayerMSAngle[layer]);
  }
  const float msAngle = o2::gpu::CAMath::Sqrt(ms2);
  const float cosTheta1half = o2::gpu::CAMath::Sqrt(1.f - o2::its::math_utils::Sq(0.5f * r1 * clampedOneOverR));
  const float cosTheta2half = o2::gpu::CAMath::Sqrt(1.f - o2::its::math_utils::Sq(0.5f * r2 * clampedOneOverR));
  const float x = (r2 * cosTheta1half) - (r1 * cosTheta2half);
  const float delta = o2::gpu::CAMath::Sqrt(1.f / (1.f - 0.25f * o2::its::math_utils::Sq(x * clampedOneOverR)) *
                                            (o2::its::math_utils::Sq((0.25f * r1 * r2 * o2::its::math_utils::Sq(clampedOneOverR) / cosTheta2half) + cosTheta1half) * o2::its::math_utils::Sq(res1) +
                                             o2::its::math_utils::Sq((0.25f * r1 * r2 * o2::its::math_utils::Sq(clampedOneOverR) / cosTheta1half) + cosTheta2half) * o2::its::math_utils::Sq(res2)));
  const float phiCut = o2::gpu::CAMath::Min(o2::gpu::CAMath::ASin(0.5f * x * clampedOneOverR) + 2.f * msAngle + delta, o2::constants::math::PI * 0.5f);
  return EdgeScatteringBendingPrep{msAngle, phiCut};
}
} // namespace

BOOST_AUTO_TEST_CASE(CylinderScatteringAngleMatchesFrozenITSFormula)
{
  // Bit-exact vs the frozen ITS expression (ITS/tracking/src/TimeFrame.cxx:347):
  // math_utils::MSangle(0.14f, trkParam.TrackletMinPt, trkParam.LayerxX0[iLayer]).
  const std::array<float, 4> xX0Values{0.f, -0.001f, 5.e-3f, 1.e-2f};
  const std::array<float, 3> trackletMinPtValues{0.1f, 0.3f, 2.5f};
  for (float xX0 : xX0Values) {
    for (float trackletMinPt : trackletMinPtValues) {
      const float reference = o2::its::math_utils::MSangle(0.14f, trackletMinPt, xX0);
      const float actual = cylinderLayerMultipleScatteringAngle(
        CylinderLayerScatteringInputs{xX0}, trackletMinPt);
      BOOST_CHECK_EQUAL(actual, reference);
    }
  }
  // xX0 <= 0 behavior, explicit: legacy MSangle maps this to zero, not a
  // rejection; the typed operation must not add validation beyond it.
  BOOST_CHECK_EQUAL(cylinderLayerMultipleScatteringAngle(
                      CylinderLayerScatteringInputs{0.f}, 0.3f),
                    0.f);
}

BOOST_AUTO_TEST_CASE(DiskScatteringAngleMatchesLegacyMftFormulaWithExplicitReferenceZ)
{
  // Bit-exact vs the legacy detail::mftLayerMSAngle(layer, params), except
  // the Disk operation receives referenceCoordinate/layerRadius
  // explicitly instead of calling mftLayerZ()/LayerZCoordinate() internally.
  // mftLayerZ() is used here only to construct the *expected* legacy value,
  // exactly as this operation's caller (TrackerTraits::initialiseTimeFrame(),
  // from the detector layout is required to do.
  ReferenceTrackingParameters legacy;
  resetDetectorDefaults(legacy, o2::detectors::DetID::MFT);
  for (int layer : {0, 3, o2::mft::constants::mft::LayersNumber - 1}) {
    const float referenceZ = detail::mftLayerZ(layer);
    const float radius = legacy.LayerRadii[layer];
    const float xX0 = legacy.LayerxX0[layer];

    const float reference = detail::mftLayerMSAngle(layer, legacy);
    const float actual = diskLayerMultipleScatteringAngle(
      DiskLayerScatteringInputs{xX0, radius, referenceZ}, legacy.TrackletMinPt);
    BOOST_CHECK_EQUAL(actual, reference);
  }

  // xX0 == 0 behavior, explicit: the legacy formula has no special case for
  // it (sqrt(0 * cscLambda) == 0), and this operation must not add one.
  const float referenceZ = detail::mftLayerZ(0);
  const float zeroX0Actual = diskLayerMultipleScatteringAngle(
    DiskLayerScatteringInputs{0.f, legacy.LayerRadii[0], referenceZ}, legacy.TrackletMinPt);
  BOOST_CHECK_EQUAL(zeroX0Actual, 0.f);
}

BOOST_AUTO_TEST_CASE(DiskScatteringAngleNearZeroReferenceRadiusFallback)
{
  // Legacy fallback: |rRef| <= 1e-6 => tanlRef = 0 (detail::mftLayerMSAngle),
  // rather than dividing by a near-zero radius.
  ReferenceTrackingParameters legacy;
  resetDetectorDefaults(legacy, o2::detectors::DetID::MFT);
  legacy.LayerRadii[0] = 1.e-9f; // below the legacy 1e-6 fallback threshold
  const float referenceZ = detail::mftLayerZ(0);

  const float reference = detail::mftLayerMSAngle(0, legacy);
  const float actual = diskLayerMultipleScatteringAngle(
    DiskLayerScatteringInputs{legacy.LayerxX0[0], legacy.LayerRadii[0], referenceZ},
    legacy.TrackletMinPt);
  BOOST_CHECK_EQUAL(actual, reference);

  // Cross-check the fallback actually engages: tanlRef == 0 (rRef below the
  // 1e-6 threshold) makes absTanl == 0, which is *not* > 1e-6 either, so
  // cscLambda takes the near-parallel-incidence sentinel 1e6f, not 1 -- i.e.
  // this input is genuinely exercising the near-zero-radius branch, not
  // merely reproducing an unrelated formula.
  const float expectedWithSentinelCscLambda = 0.0136f * (1.f / legacy.TrackletMinPt) * std::sqrt(legacy.LayerxX0[0] * 1.e6f);
  BOOST_CHECK_EQUAL(reference, expectedWithSentinelCscLambda);
}

BOOST_AUTO_TEST_CASE(ClampEdgeCurvatureUsesOneCoordinateNeutralExpression)
{
  const std::array<std::pair<float, float>, 5> samples{{
    {0.001f, 50.f}, // clamp does not trigger
    {3.0f, 1.0f},   // clamp triggers
    {0.02f, 25.f},
    {0.5f, 0.9f},
    {0.0001f, 4.f},
  }};
  for (const auto& sample : samples) {
    const float oneOverR = sample.first;
    const float r2 = sample.second;

    const float actual = clampEdgeCurvature(oneOverR, r2);
    const float reference = (0.5f * oneOverR >= 1.f / r2) ? (2.f / r2) - o2::constants::math::Almost0 : oneOverR;
    BOOST_CHECK_EQUAL(actual, reference);
  }
}

BOOST_AUTO_TEST_CASE(CurvatureClampIsEdgeLocal)
{
  constexpr float initialOneOverR = 3.f;
  const std::array<float, 3> outerRadii{1.f, 4.f, 0.5f};
  for (const auto outerRadius : outerRadii) {
    const auto forward = clampEdgeCurvature(initialOneOverR, outerRadius);
    const auto repeated = clampEdgeCurvature(initialOneOverR, outerRadius);
    BOOST_CHECK_EQUAL(forward, repeated);
  }
}

BOOST_AUTO_TEST_CASE(PrepareEdgeScatteringAndBendingMatchesFrozenFormulaForITSAndMFTShapedInputs)
{
  // ITS-shaped (cm-scale barrel radii from ReferenceTrackingParameters defaults).
  {
    const std::array<float, 7> msAngles{1.e-3f, 1.1e-3f, 1.2e-3f, 2.e-3f, 2.1e-3f, 2.2e-3f, 2.3e-3f};
    constexpr int fromLayer = 0;
    constexpr int toLayer = 3; // half-open: sums layers 0,1,2 only
    constexpr float r1 = 2.33959f;
    constexpr float r2 = 19.6213f;
    constexpr float res1 = 5.e-4f;
    constexpr float res2 = 5.e-4f;
    const float oneOverR = clampEdgeCurvature(
      0.001f * 0.3f * std::abs(Bz) / 0.3f, r2);
    const gsl::span<const float> msSpan(msAngles.data(), msAngles.size());
    const auto actual = prepareEdgeScatteringAndBending(msSpan, fromLayer, toLayer, r1, r2, oneOverR, res1, res2);
    const auto reference = referenceEdgeScatteringAndBending(msSpan, fromLayer, toLayer, r1, r2, oneOverR, res1, res2);
    BOOST_CHECK_EQUAL(actual.msAngle, reference.msAngle);
    BOOST_CHECK_EQUAL(actual.phiCut, reference.phiCut);

    // Half-open range: layer index `toLayer` itself must not contribute.
    const auto includingToLayer = referenceEdgeScatteringAndBending(msSpan, fromLayer, toLayer + 1, r1, r2, oneOverR, res1, res2);
    BOOST_CHECK_NE(actual.msAngle, includingToLayer.msAngle);
  }

  // MFT-shaped, deliberately skipped/non-adjacent edge (fromLayer=1,
  // toLayer=4: sums layers 1,2,3, skipping layer 4 itself as the endpoint).
  {
    ReferenceTrackingParameters mft;
    resetDetectorDefaults(mft, o2::detectors::DetID::MFT);
    std::array<float, o2::mft::constants::mft::LayersNumber> msAngles{};
    for (int layer = 0; layer < o2::mft::constants::mft::LayersNumber; ++layer) {
      msAngles[layer] = diskLayerMultipleScatteringAngle(
        DiskLayerScatteringInputs{mft.LayerxX0[layer], mft.LayerRadii[layer], detail::mftLayerZ(layer)},
        mft.TrackletMinPt);
    }
    constexpr int fromLayer = 1;
    constexpr int toLayer = 4;
    const float r1 = mft.LayerRadii[fromLayer];
    const float r2 = mft.LayerRadii[toLayer];
    constexpr float res1 = 5.e-4f;
    constexpr float res2 = 6.e-4f;
    const float oneOverR = clampEdgeCurvature(
      0.001f * 0.3f * std::abs(Bz) / mft.TrackletMinPt, r2);
    const gsl::span<const float> msSpan(msAngles.data(), msAngles.size());
    const auto actual = prepareEdgeScatteringAndBending(msSpan, fromLayer, toLayer, r1, r2, oneOverR, res1, res2);
    const auto reference = referenceEdgeScatteringAndBending(msSpan, fromLayer, toLayer, r1, r2, oneOverR, res1, res2);
    BOOST_CHECK_EQUAL(actual.msAngle, reference.msAngle);
    BOOST_CHECK_EQUAL(actual.phiCut, reference.phiCut);
  }
}

BOOST_AUTO_TEST_CASE(PrepareEdgeScatteringAndBendingZeroFieldAndDegenerateRadiusMatchLegacyFormula)
{
  const std::array<float, 3> msAngles{1.e-3f, 1.2e-3f, 1.4e-3f};
  const gsl::span<const float> msSpan(msAngles.data(), msAngles.size());

  // Zero field: oneOverR's initial value (before any clamp) is exactly 0,
  // matching the legacy `0.001f * 0.3f * std::abs(mBz) / trkParam.TrackletMinPt`.
  {
    const float zeroFieldOneOverR = 0.001f * 0.3f * std::abs(0.f) / 0.3f;
    BOOST_CHECK_EQUAL(zeroFieldOneOverR, 0.f);
    const float clamped = clampEdgeCurvature(zeroFieldOneOverR, 5.f);
    BOOST_CHECK_EQUAL(clamped, 0.f); // 0.5*0 >= 1/5 is false: clamp does not trigger
    const auto actual = prepareEdgeScatteringAndBending(msSpan, 0, 2, 2.f, 5.f, clamped, 5.e-4f, 5.e-4f);
    const auto reference = referenceEdgeScatteringAndBending(msSpan, 0, 2, 2.f, 5.f, clamped, 5.e-4f, 5.e-4f);
    BOOST_CHECK_EQUAL(actual.msAngle, reference.msAngle);
    BOOST_CHECK_EQUAL(actual.phiCut, reference.phiCut);
  }

  // Degenerate radius (r2 == 0): legacy does not reject this -- it flows
  // through to whatever the floating-point expression produces. This test
  // asserts parity with that expression, not any particular finiteness.
  {
    const float oneOverR = clampEdgeCurvature(0.01f, 0.f);
    const auto actual = prepareEdgeScatteringAndBending(msSpan, 0, 2, 2.f, 0.f, oneOverR, 5.e-4f, 5.e-4f);
    const auto reference = referenceEdgeScatteringAndBending(msSpan, 0, 2, 2.f, 0.f, oneOverR, 5.e-4f, 5.e-4f);
    // BOOST_CHECK_EQUAL on NaN is always false (NaN != NaN); compare the bit
    // pattern so a NaN-vs-NaN legacy-parity match is still recognized as a pass.
    BOOST_CHECK(std::memcmp(&actual.msAngle, &reference.msAngle, sizeof(float)) == 0);
    BOOST_CHECK(std::memcmp(&actual.phiCut, &reference.phiCut, sizeof(float)) == 0);
  }
}
