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

#ifndef ALICEO2_ITSMFT_TRACKING_PARAMETER_TEST_SUPPORT_H_
#define ALICEO2_ITSMFT_TRACKING_PARAMETER_TEST_SUPPORT_H_
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/detail/MFTFwdTrackHelpers.h"

namespace o2::itsmft::tracking::test
{
template <typename T>
concept HasDetectorRadii = requires(T value) { value.LayerRadii; };
template <typename T>
concept HasMemoryPolicy = requires(T value) { value.MaxMemory; };
template <typename T>
concept HasFailurePolicy = requires(T value) { value.DropTFUponFailure; };
static_assert(!HasDetectorRadii<IterationParameters>);
static_assert(!HasMemoryPolicy<IterationParameters>);
static_assert(!HasFailurePolicy<IterationParameters>);

// Retain the old input shape only for independent numerical reference fixtures.
struct ReferenceTrackingParameters : TrackingParameters {
  std::vector<float> LayerxX0{kNominalITSLayerX0.begin(), kNominalITSLayerX0.end()};
};
inline void resetDetectorDefaults(ReferenceTrackingParameters& parameters, o2::detectors::DetID::ID detector)
{
  o2::itsmft::resetDetectorDefaults(parameters, detector);
  parameters.LayerxX0.clear();
  const auto catalog = detector == o2::detectors::DetID::ITS
                         ? SurfaceCatalogView{kITSStaticSurfaceCatalog.data(), kITSStaticSurfaceCatalog.size()}
                         : SurfaceCatalogView{kMFTStaticSurfaceCatalog.data(), kMFTStaticSurfaceCatalog.size()};
  for (uint32_t layer = 0; layer < catalog.nSurfaces; ++layer) {
    parameters.LayerxX0.push_back(catalog.surfaces[layer].material.xOverX0);
  }
}
inline TrackingPlan makeTrackingPlan(const TrackingParameters& parameters)
{
  return {parameters, parameters, {parameters}};
}
inline TrackingPlan makeTrackingPlan(TrackingParameters&& parameters)
{
  TrackingPlan plan{std::move(static_cast<DetectorParameters&>(parameters)), parameters, {}};
  plan.iterations.push_back(std::move(static_cast<IterationParameters&>(parameters)));
  return plan;
}
template <typename Parameters>
TrackingPlan makeTrackingPlan(const std::vector<Parameters>& parameters)
{
  if (parameters.empty()) {
    return {};
  }
  auto plan = makeTrackingPlan(parameters.front());
  plan.iterations.assign(parameters.begin(), parameters.end());
  return plan;
}
// Expand the split result solely to keep pre-refactor preset assertions intact.
inline std::vector<TrackingParameters> expandTrackingPlan(const TrackingPlan& plan)
{
  std::vector<TrackingParameters> result;
  for (const auto& iteration : plan.iterations) {
    result.push_back({iteration, plan.detector, plan.execution});
  }
  return result;
}
inline std::vector<TrackingParameters> referenceTrackingParameters(o2::detectors::DetID::ID detector, TrackingMode::Type mode)
{
  return expandTrackingPlan(TrackingMode::getTrackingPlan(detector, mode));
}
} // namespace o2::itsmft::tracking::test
namespace o2::itsmft::tracking::detail
{
inline float mftLayerMSAngle(int layer, const test::ReferenceTrackingParameters& params)
{
  const float invP = 1.f / params.TrackletMinPt;
  const float zLayer = mftLayerZ(layer);
  const float rRef = params.LayerRadii[layer];
  const float tanlRef = (std::abs(rRef) > 1e-6f) ? zLayer / rRef : 0.f;
  const float absTanl = std::abs(tanlRef);
  const float cscLambda = (absTanl > 1e-6f) ? std::sqrt(1.f + tanlRef * tanlRef) / absTanl : 1e6f;
  return 0.0136f * invP * std::sqrt(params.LayerxX0[layer] * cscLambda);
}

} // namespace o2::itsmft::tracking::detail
#endif
