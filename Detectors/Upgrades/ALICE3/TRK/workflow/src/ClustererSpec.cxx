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

namespace o2::trk
{

void ClustererDPL::init(o2::framework::InitContext& ic)
{
  mNThreads = std::max(1, ic.options().get<int>("nthreads"));
}

void ClustererDPL::run(o2::framework::ProcessingContext& pc)
{
  auto digits = pc.inputs().get<gsl::span<o2::itsmft::Digit>>("digits");
  auto rofs = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");

  gsl::span<const o2::itsmft::MC2ROFRecord> mc2rofs;
  gsl::span<const char> labelbuffer;
  if (mUseMC) {
    labelbuffer = pc.inputs().get<gsl::span<char>>("labels");
    mc2rofs = pc.inputs().get<gsl::span<o2::itsmft::MC2ROFRecord>>("MC2ROframes");
  }
  o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelbuffer);

  std::vector<o2::trk::Cluster> clusters;
  std::vector<unsigned char> patterns;
  std::vector<o2::trk::ROFRecord> clusterROFs;
  std::unique_ptr<o2::dataformats::MCTruthContainer<o2::MCCompLabel>> clusterLabels;
  std::vector<o2::trk::MC2ROFRecord> clusterMC2ROFs;
  if (mUseMC) {
    clusterLabels = std::make_unique<o2::dataformats::MCTruthContainer<o2::MCCompLabel>>();
  }
  o2::base::GeometryManager::loadGeometry("o2sim_geometry.root", false, true);

  mClusterer.process(digits,
                     rofs,
                     clusters,
                     patterns,
                     clusterROFs,
                     mUseMC ? &labels : nullptr,
                     clusterLabels.get(),
                     mc2rofs,
                     mUseMC ? &clusterMC2ROFs : nullptr);

  pc.outputs().snapshot(o2::framework::Output{"TRK", "COMPCLUSTERS", 0}, clusters);
  pc.outputs().snapshot(o2::framework::Output{"TRK", "PATTERNS", 0}, patterns);
  pc.outputs().snapshot(o2::framework::Output{"TRK", "CLUSTERSROF", 0}, clusterROFs);

  if (mUseMC) {
    pc.outputs().snapshot(o2::framework::Output{"TRK", "CLUSTERSMCTR", 0}, *clusterLabels);
    pc.outputs().snapshot(o2::framework::Output{"TRK", "CLUSTERSMC2ROF", 0}, clusterMC2ROFs);
  }

  LOGP(info, "TRKClusterer pushed {} clusters in {} ROFs", clusters.size(), clusterROFs.size());
}

o2::framework::DataProcessorSpec getClustererSpec(bool useMC)
{
  std::vector<o2::framework::InputSpec> inputs;
  inputs.emplace_back("digits", "TRK", "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "TRK", "DIGITSROF", 0, o2::framework::Lifetime::Timeframe);

  std::vector<o2::framework::OutputSpec> outputs;
  outputs.emplace_back("TRK", "COMPCLUSTERS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("TRK", "PATTERNS", 0, o2::framework::Lifetime::Timeframe);
  outputs.emplace_back("TRK", "CLUSTERSROF", 0, o2::framework::Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labels", "TRK", "DIGITSMCTR", 0, o2::framework::Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "TRK", "DIGITSMC2ROF", 0, o2::framework::Lifetime::Timeframe);
    outputs.emplace_back("TRK", "CLUSTERSMCTR", 0, o2::framework::Lifetime::Timeframe);
    outputs.emplace_back("TRK", "CLUSTERSMC2ROF", 0, o2::framework::Lifetime::Timeframe);
  }

  return o2::framework::DataProcessorSpec{
    "trk-clusterer",
    inputs,
    outputs,
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<o2::trk::ClustererDPL>(useMC)},
    o2::framework::Options{{"nthreads", o2::framework::VariantType::Int, 1, {"Number of clustering threads"}}}};
}

} // namespace o2::trk
