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
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/TimeFrame.h"
#include "ITSMFTTracking/ROFLookupTables.h"
#include <functional>
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"

namespace o2::itsmft::tracking::test
{
template <typename T>
concept HasDetectorRadii = requires(T value) { value.LayerRadii; };
template <typename T>
concept HasMemoryPolicy = requires(T value) { value.MaxMemory; };
template <typename T>
concept HasFailurePolicy = requires(T value) { value.DropTFUponFailure; };
static_assert(!HasDetectorRadii<IterationParameters>);
static_assert(!HasDetectorRadii<DetectorParameters>);
static_assert(!HasMemoryPolicy<IterationParameters>);
static_assert(!HasFailurePolicy<IterationParameters>);

// Retain the old input shape only for independent numerical reference fixtures.
struct ReferenceTrackingParameters : TrackingParameters {
  // Frozen pre-consolidation radii for independent numerical oracles.
  std::vector<float> LayerRadii = {2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
  std::vector<float> LayerxX0 = {5.e-3f, 5.e-3f, 5.e-3f, 1.e-2f, 1.e-2f, 1.e-2f, 1.e-2f};
};
inline void resetDetectorDefaults(ReferenceTrackingParameters& parameters, o2::detectors::DetID::ID detector)
{
  o2::itsmft::resetDetectorDefaults(parameters, detector);
  parameters.LayerRadii = ReferenceTrackingParameters{}.LayerRadii;
  if (detector == o2::detectors::DetID::MFT) {
    constexpr std::array<float, MFTNLayers> minima{2.1f, 2.1f, 2.1f, 2.1f, 2.1f, 2.1f, 3.1f, 3.1f, 3.5f, 3.5f};
    constexpr std::array<float, MFTNLayers> maxima{12.5f, 12.5f, 12.5f, 12.5f, 14.f, 14.f, 17.f, 17.f, 17.5f, 17.5f};
    parameters.LayerRadii.resize(MFTNLayers);
    for (int layer = 0; layer < MFTNLayers; ++layer) {
      parameters.LayerRadii[layer] = 0.5f * (minima[layer] + maxima[layer]);
    }
  }
  parameters.LayerxX0.clear();
  const auto catalog = detector == o2::detectors::DetID::ITS
                         ? SurfaceCatalogView{kITSSurfaces.data(), kITSSurfaces.size()}
                         : SurfaceCatalogView{kMFTSurfaces.data(), kMFTSurfaces.size()};
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
// Synthetic decoding is confined to tests. Exercise the same normalization
// and ROF bookkeeping as production without constructing detector geometry.
struct TestClusterSourceInput : ClusterSourceInput {
  // Fixture-owned timing is bound separately after cluster loading.
  o2::its::LayerTiming timing{};
  RuntimeROFViews rofViews{};
  std::function<DecodedCluster(const itsmft::CompClusterExt&, gsl::span<const unsigned char>::iterator&,
                               const itsmft::TopologyDictionary*, uint32_t)>
    decode;

  template <typename Decoder>
  void setDecoder(const Decoder& decoder)
  {
    decode = [&decoder](const auto& cluster, auto& patterns, const auto* dictionary, uint32_t index) {
      return decoder.decode(cluster, patterns, dictionary, index);
    };
  }
};

inline void loadSources(TimeFrame& frame, const SurfaceCatalogView& catalog,
                        gsl::span<const TestClusterSourceInput> sources, const o2::InteractionRecord&,
                        std::vector<std::vector<uint32_t>>* indices = nullptr,
                        std::vector<std::vector<uint32_t>>* sizes = nullptr, bool requireCompleteMapping = false)
{
  const std::vector<ClusterSourceInput> inputs(sources.begin(), sources.end());
  detail::prepareSources(frame, catalog, inputs, indices, sizes, requireCompleteMapping);
  std::vector<std::vector<uint32_t>> externalIndices(catalog.nSurfaces);
  std::vector<std::vector<uint32_t>> clusterSizes(catalog.nSurfaces);
  bool hasMCInformation = false;
  for (const auto& source : sources) {
    detail::validateClusterRanges(source);
    detail::loadDecodedSource(frame, catalog, source, [&](const auto& cluster, auto& patterns) {
      const auto index = static_cast<uint32_t>(&cluster - source.clusters.data());
      return source.decode(cluster, patterns, source.dictionary, index); }, externalIndices, clusterSizes);
    hasMCInformation |= source.labels != nullptr;
  }
  frame.setHasMCInformation(hasMCInformation);
  if (!sources.empty()) {
    frame.setROFViews(sources.front().rofViews);
    for (const auto& source : sources) {
      for (uint16_t layer = 0; layer < source.layerToSurface.size(); ++layer) {
        frame.setROFViews(source.layerToSurface[layer].value(), source.rofViews, layer);
      }
    }
  }
  if (indices != nullptr) {
    *indices = std::move(externalIndices);
  }
  if (sizes != nullptr) {
    *sizes = std::move(clusterSizes);
  }
}

inline void loadTimeFrameSources(TimeFrame& frame, gsl::span<const TestClusterSourceInput> sources,
                                 SurfaceCatalogView catalog, const o2::InteractionRecord& origin,
                                 std::vector<std::vector<uint32_t>>* indices = nullptr,
                                 std::vector<std::vector<uint32_t>>* sizes = nullptr)
{
  loadSources(frame, catalog, sources, origin, indices, sizes, true);
}

template <typename Decoder>
void loadTimeFrameSource(
  TimeFrame& frame,
  const Decoder& decoder,
  const o2::InteractionRecord& origin,
  const o2::its::LayerTiming& timing,
  gsl::span<const itsmft::CompClusterExt> clusters,
  gsl::span<const unsigned char> patterns,
  gsl::span<const o2::itsmft::ROFRecord> rofs,
  const itsmft::TopologyDictionary* dictionary,
  const dataformats::MCTruthContainer<MCCompLabel>* labels,
  o2::detectors::DetID::ID detector,
  gsl::span<const LayerId> layerToSurface,
  SurfaceCatalogView catalog,
  std::vector<std::vector<uint32_t>>* externalIndicesBySurface = nullptr,
  std::vector<std::vector<uint32_t>>* clusterSizesBySurface = nullptr)
{
  constexpr ClusterSourceId sourceId{0};
  TestClusterSourceInput source;
  source.id = sourceId;
  source.detector = detector;
  source.clusters = clusters;
  source.patterns = patterns;
  source.rofs = rofs;
  source.dictionary = dictionary;
  source.labels = labels;
  source.layerToSurface = layerToSurface;
  source.timing = timing;
  source.setDecoder(decoder);
  source.rofViews = frame.getROFViews();
  loadTimeFrameSources(frame, gsl::span<const TestClusterSourceInput>{&source, 1}, catalog, origin,
                       externalIndicesBySurface, clusterSizesBySurface);
}

} // namespace o2::itsmft::tracking::test
#endif
