// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRKWorkflow/ClustererSpec.h"
#include "TRKBase/AlmiraParam.h"
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsTRKFT3/Cluster.h"
#include "DataFormatsTRKFT3/ROFRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <string>

namespace o2::trk
{

template <int DetID>
void ClustererDPL<DetID>::init(o2::framework::InitContext& ic)
{
  mNThreads = std::max(1, ic.options().get<int>("nthreads"));
#ifdef O2_WITH_ACTS
  if constexpr (DetID == o2::detectors::DetID::TRK) {
    mUseACTS = ic.options().get<bool>("useACTS");
  }
#endif
}

template <int DetID>
void ClustererDPL<DetID>::run(o2::framework::ProcessingContext& pc)
{
  o2::base::GeometryManager::loadGeometry(); // default prefix -> o2sim_geometry[-aligned].root

  constexpr auto Origin = DetID == o2::detectors::DetID::TRK ? o2::header::gDataOriginTRK : o2::header::gDataOriginFT3;
  const auto detName = Origin.template as<std::string>();
  uint64_t totalClusters = 0;
  constexpr int NLayers = DetID == o2::detectors::DetID::TRK ? static_cast<int>(o2::trk::AlmiraParam::kNLayers) : o2::trk::constants::MLOTDisks::nLayers;
  for (int iLayer = 0; iLayer < NLayers; ++iLayer) {
    auto digits = pc.inputs().get<gsl::span<o2::trkft3::Digit>>(std::format("digits_{}", iLayer));
    auto rofs = pc.inputs().get<gsl::span<o2::trkft3::ROFRecord>>(std::format("ROframes_{}", iLayer));

    gsl::span<const char> labelbuffer;
    if (mUseMC) {
      labelbuffer = pc.inputs().get<gsl::span<char>>(std::format("labels_{}", iLayer));
    }
    o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelbuffer);

    std::vector<o2::trkft3::Cluster<DetID>> clusters;
    std::vector<unsigned char> patterns;
    std::vector<o2::trkft3::ROFRecord> clusterROFs;
    std::unique_ptr<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> clusterLabels;
    if (mUseMC) {
      clusterLabels = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
    }

    bool clustered = false;
#ifdef O2_WITH_ACTS
    if constexpr (DetID == o2::detectors::DetID::TRK) {
      if (mUseACTS) {
        LOG(info) << "Running TRKClusterer with ACTS on layer " << iLayer;
        mClustererACTS.process(digits,
                               rofs,
                               clusters,
                               patterns,
                               clusterROFs,
                               mUseMC ? &labels : nullptr,
                               clusterLabels.get());
        clustered = true;
      }
    }
#endif
    if (!clustered) {
      LOG(info) << "Running " << detName << "Clusterer on layer " << iLayer;
      mClusterer.process(digits,
                         rofs,
                         clusters,
                         patterns,
                         clusterROFs,
                         mUseMC ? &labels : nullptr,
                         clusterLabels.get());
    }

    const auto subspec = static_cast<o2::framework::DataAllocator::SubSpecificationType>(iLayer);
    pc.outputs().snapshot(o2::framework::Output{Origin, "COMPCLUSTERS", subspec}, clusters);
    pc.outputs().snapshot(o2::framework::Output{Origin, "PATTERNS", subspec}, patterns);
    pc.outputs().snapshot(o2::framework::Output{Origin, "CLUSTERSROF", subspec}, clusterROFs);
    if (mUseMC) {
      pc.outputs().snapshot(o2::framework::Output{Origin, "CLUSTERSMCTR", subspec}, *clusterLabels);
    }
    totalClusters += clusters.size();
    LOGP(info, "{}Clusterer layer {} pushed {} clusters in {} ROFs", detName, iLayer, clusters.size(), clusterROFs.size());
  }

  LOGP(info, "{}Clusterer produced {} clusters", detName, totalClusters);
}

namespace
{
template <int DetID>
o2::framework::DataProcessorSpec getClustererSpecT(bool useMC)
{
  constexpr auto origin = DetID == o2::detectors::DetID::TRK ? o2::header::gDataOriginTRK : o2::header::gDataOriginFT3;
  std::vector<o2::framework::InputSpec> inputs;
  constexpr int NLayers = DetID == o2::detectors::DetID::TRK ? static_cast<int>(o2::trk::AlmiraParam::kNLayers) : o2::trk::constants::MLOTDisks::nLayers;
  for (int iLayer = 0; iLayer < NLayers; ++iLayer) {
    inputs.emplace_back(std::format("digits_{}", iLayer), origin, "DIGITS", iLayer, o2::framework::Lifetime::Timeframe);
    inputs.emplace_back(std::format("ROframes_{}", iLayer), origin, "DIGITSROF", iLayer, o2::framework::Lifetime::Timeframe);
    if (useMC) {
      inputs.emplace_back(std::format("labels_{}", iLayer), origin, "DIGITSMCTR", iLayer, o2::framework::Lifetime::Timeframe);
    }
  }

  std::vector<o2::framework::OutputSpec> outputs;
  for (int iLayer = 0; iLayer < NLayers; ++iLayer) {
    outputs.emplace_back(origin, "COMPCLUSTERS", iLayer, o2::framework::Lifetime::Timeframe);
    outputs.emplace_back(origin, "PATTERNS", iLayer, o2::framework::Lifetime::Timeframe);
    outputs.emplace_back(origin, "CLUSTERSROF", iLayer, o2::framework::Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back(origin, "CLUSTERSMCTR", iLayer, o2::framework::Lifetime::Timeframe);
    }
  }

  o2::framework::Options options{{"nthreads", o2::framework::VariantType::Int, 1, {"Number of clustering threads"}}};
#ifdef O2_WITH_ACTS
  if constexpr (DetID == o2::detectors::DetID::TRK) {
    options.push_back({"useACTS", o2::framework::VariantType::Bool, false, {"Use ACTS for clustering"}});
  }
#endif
  auto name = origin.template as<std::string>() + "-clusterer";
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
  return o2::framework::DataProcessorSpec{
    name,
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<o2::trk::ClustererDPL<DetID>>(useMC)},
    options};
}
} // namespace

o2::framework::DataProcessorSpec getClustererSpec(bool useMC)
{
  return getClustererSpecT<o2::detectors::DetID::TRK>(useMC);
}

o2::framework::DataProcessorSpec getFT3ClustererSpec(bool useMC)
{
  return getClustererSpecT<o2::detectors::DetID::FT3>(useMC);
}

template class ClustererDPL<o2::detectors::DetID::TRK>;
template class ClustererDPL<o2::detectors::DetID::FT3>;

} // namespace o2::trk
