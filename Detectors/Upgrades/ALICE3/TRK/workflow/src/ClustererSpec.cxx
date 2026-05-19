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
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsTRK/Cluster.h"
#include "DataFormatsTRK/ROFRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"

#include <format>

namespace o2::trk
{

void ClustererDPL::init(o2::framework::InitContext& ic)
{
  mNThreads = std::max(1, ic.options().get<int>("nthreads"));
#ifdef O2_WITH_ACTS
  mUseACTS = ic.options().get<bool>("useACTS");
#endif
}

void ClustererDPL::run(o2::framework::ProcessingContext& pc)
{
  o2::base::GeometryManager::loadGeometry("sgn_geometry.root", false, true);

  uint64_t totalClusters = 0;
  for (int iLayer = 0; iLayer < mLayers; ++iLayer) {
    auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>(std::format("digits_{}", iLayer));
    auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>(std::format("ROframes_{}", iLayer));

    gsl::span<const char> labelbuffer;
    if (mUseMC) {
      labelbuffer = pc.inputs().get<gsl::span<char>>(std::format("labels_{}", iLayer));
    }
    o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelbuffer);

    std::vector<o2::trk::Cluster> clusters;
    std::vector<unsigned char> patterns;
    std::vector<o2::trk::ROFRecord> clusterROFs;
    std::unique_ptr<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> clusterLabels;
    if (mUseMC) {
      clusterLabels = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
    }

#ifdef O2_WITH_ACTS
    if (mUseACTS) {
      LOG(info) << "Running TRKClusterer with ACTS on layer " << iLayer;
      mClustererACTS.process(digits,
                             rofs,
                             clusters,
                             patterns,
                             clusterROFs,
                             mUseMC ? &labels : nullptr,
                             clusterLabels.get());
    } else
#endif
    {
      LOG(info) << "Running TRKClusterer on layer " << iLayer;
      mClusterer.process(digits,
                         rofs,
                         clusters,
                         patterns,
                         clusterROFs,
                         mUseMC ? &labels : nullptr,
                         clusterLabels.get());
    }

    const auto subspec = static_cast<o2::framework::DataAllocator::SubSpecificationType>(iLayer);
    pc.outputs().snapshot(o2::framework::Output{"TRK", "COMPCLUSTERS", subspec}, clusters);
    pc.outputs().snapshot(o2::framework::Output{"TRK", "PATTERNS", subspec}, patterns);
    pc.outputs().snapshot(o2::framework::Output{"TRK", "CLUSTERSROF", subspec}, clusterROFs);
    if (mUseMC) {
      pc.outputs().snapshot(o2::framework::Output{"TRK", "CLUSTERSMCTR", subspec}, *clusterLabels);
    }
    totalClusters += clusters.size();
    LOGP(info, "TRKClusterer layer {} pushed {} clusters in {} ROFs", iLayer, clusters.size(), clusterROFs.size());
  }

  LOGP(info, "TRKClusterer produced {} clusters", totalClusters);
}

o2::framework::DataProcessorSpec getClustererSpec(bool useMC)
{
  static constexpr int nLayers = o2::trk::AlmiraParam::kNLayers;
  std::vector<o2::framework::InputSpec> inputs;
  for (int iLayer = 0; iLayer < nLayers; ++iLayer) {
    inputs.emplace_back(std::format("digits_{}", iLayer), "TRK", "DIGITS", iLayer, o2::framework::Lifetime::Timeframe);
    inputs.emplace_back(std::format("ROframes_{}", iLayer), "TRK", "DIGITSROF", iLayer, o2::framework::Lifetime::Timeframe);
    if (useMC) {
      inputs.emplace_back(std::format("labels_{}", iLayer), "TRK", "DIGITSMCTR", iLayer, o2::framework::Lifetime::Timeframe);
    }
  }

  std::vector<o2::framework::OutputSpec> outputs;
  for (int iLayer = 0; iLayer < nLayers; ++iLayer) {
    outputs.emplace_back("TRK", "COMPCLUSTERS", iLayer, o2::framework::Lifetime::Timeframe);
    outputs.emplace_back("TRK", "PATTERNS", iLayer, o2::framework::Lifetime::Timeframe);
    outputs.emplace_back("TRK", "CLUSTERSROF", iLayer, o2::framework::Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back("TRK", "CLUSTERSMCTR", iLayer, o2::framework::Lifetime::Timeframe);
    }
  }

  return o2::framework::DataProcessorSpec{
    "trk-clusterer",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<o2::trk::ClustererDPL>(useMC)},
    o2::framework::Options{{"nthreads", o2::framework::VariantType::Int, 1, {"Number of clustering threads"}}
#ifdef O2_WITH_ACTS
                           ,
                           {"useACTS", o2::framework::VariantType::Bool, false, {"Use ACTS for clustering"}}
#endif
    }};
}

} // namespace o2::trk
