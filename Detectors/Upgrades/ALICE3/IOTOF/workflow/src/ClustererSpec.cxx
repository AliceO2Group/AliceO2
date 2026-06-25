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

#include "IOTOFWorkflow/ClustererSpec.h"
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsIOTOF/Cluster.h"
#include "DataFormatsIOTOF/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"

#include <format>

using namespace o2::framework;

namespace o2::iotof
{

void ClustererDPL::init(o2::framework::InitContext& ic)
{
  mNThreads = std::max(1, ic.options().get<int>("nthreads"));
}

void ClustererDPL::run(o2::framework::ProcessingContext& pc)
{
  LOG(info) << "[ClustererDPL] Entered run() with " << mNThreads << " threads";
  o2::base::GeometryManager::loadGeometry("o2sim_geometry.root", false, true);

  LOG(info) << "[ClustererDPL] Geometry loaded";
  uint64_t totalClusters = 0;
  
  // Loop on layers to be added here, for now only one layer is processed
  int iLayer = 0;
  LOG(info) << "[ClustererDPL] Getting digits for layer " << iLayer;
  auto digits = pc.inputs().get<gsl::span<o2::iotof::Digit>>(std::format("digits_{}", iLayer));
  auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>(std::format("ROframes_{}", iLayer));

  LOG(info) << "[ClustererDPL] Got " << digits.size() << " digits and " << rofs.size() << " ROFs for layer " << iLayer;
  gsl::span<const char> labelbuffer;
  if (mUseMC) {
    LOG(info) << "[ClustererDPL] Getting MC labels for layer " << iLayer;
    labelbuffer = pc.inputs().get<gsl::span<char>>(std::format("labels_{}", iLayer));
    LOG(info) << "[ClustererDPL] Got " << labelbuffer.size() << " bytes of MC labels for layer " << iLayer;
  }
  o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelbuffer);
  LOG(info) << "[ClustererDPL] Got MC labels for layer " << iLayer;

  std::vector<o2::iotof::Cluster> clusters;
  std::vector<unsigned char> patterns;
  std::vector<o2::itsmft::ROFRecord> clusterROFs;
  std::unique_ptr<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> clusterLabels;
  if (mUseMC) {
    clusterLabels = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
  }

  LOG(info) << "[ClustererDPL] Running IOTOFClusterer on layer " << iLayer;
  mClusterer.process(digits,
                     rofs,
                     clusters,
                     patterns,
                     clusterROFs,
                     mUseMC ? &labels : nullptr,
                     clusterLabels.get());
  LOG(info) << "[ClustererDPL] IOTOFClusterer produced " << clusters.size() << " clusters for layer " << iLayer;
  const auto subspec = static_cast<o2::framework::DataAllocator::SubSpecificationType>(0);
  pc.outputs().snapshot(o2::framework::Output{"TF3", "COMPCLUSTERS", subspec}, clusters);
  pc.outputs().snapshot(o2::framework::Output{"TF3", "PATTERNS", subspec}, patterns);
  pc.outputs().snapshot(o2::framework::Output{"TF3", "CLUSTERSROF", subspec}, clusterROFs);
  if (mUseMC) {
    pc.outputs().snapshot(o2::framework::Output{"TF3", "CLUSTERSMCTR", subspec}, *clusterLabels);
  }
  totalClusters += clusters.size();
  LOGP(info, "[ClustererDPL] IOTOFClusterer layer {} pushed {} clusters in {} ROFs", iLayer, clusters.size(), clusterROFs.size());

  LOGP(info, "[ClustererDPL] IOTOFClusterer produced {} clusters", totalClusters);
}

o2::framework::DataProcessorSpec getClustererSpec(bool useMC)
{

  LOG(info) << "[ClustererSpec] Creating DataProcessorSpec for IOTOFClusterer with useMC=" << useMC;
  static constexpr int nLayers = 2;
  std::vector<o2::framework::InputSpec> inputs;
  // Currently TF3 digits (unlike TRK) are not separated by layer, eventually per-layer reading here
  int iLayer = 0;
  inputs.emplace_back(std::format("digits_{}", iLayer), "TF3", "DIGITS", iLayer, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back(std::format("ROframes_{}", iLayer), "TF3", "DIGITSROF", iLayer, o2::framework::Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back(std::format("labels_{}", iLayer), "TF3", "DIGITSMCTR", iLayer, o2::framework::Lifetime::Timeframe);
  }
  LOG(info) << "[ClustererSpec] Created " << inputs.size() << " input specifications for IOTOFClusterer";

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("TF3", "COMPCLUSTERS", iLayer, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("TF3", "PATTERNS", iLayer, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("TF3", "CLUSTERSROF", iLayer, o2::framework::Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("TF3", "CLUSTERSMCTR", iLayer, o2::framework::Lifetime::Timeframe);
  }
  LOG(info) << "[ClustererSpec] Created " << outputs.size() << " output specifications for IOTOFClusterer";

  LOG(info) << "[ClustererSpec] Returning ... ";
  return o2::framework::DataProcessorSpec{
    "iotof-clusterer",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<o2::iotof::ClustererDPL>(useMC)},
    o2::framework::Options{{"nthreads", o2::framework::VariantType::Int, 1, {"Number of clustering threads"}}
    }};
}

DataProcessorSpec getIOTOFClustererSpec(bool mctruth)
{
  return getClustererSpec(mctruth);
}

} // namespace o2::iotof
