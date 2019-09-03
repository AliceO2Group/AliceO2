// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackWriterTPCITSSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ControlService.h"
#include "GlobalTrackingWorkflow/TrackWriterTPCITSSpec.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

void TrackWriterTPCITS::init(InitContext& ic)
{
  mOutFileName = ic.options().get<std::string>("tpcits-tracks-outfile");
}

void TrackWriterTPCITS::run(ProcessingContext& pc)
{
  if (mFinished) {
    return;
  }

  TFile outf(mOutFileName.c_str(), "recreate");
  if (outf.IsZombie()) {
    LOG(FATAL) << "Failed to open output file " << mOutFileName;
  }
  TTree tree(mTreeName.c_str(), "Tree of ITS-TPC matches");
  auto tracks = pc.inputs().get<const std::vector<o2::dataformats::TrackTPCITS>>("match");
  auto tracksPtr = &tracks;
  tree.Branch(mOutTPCITSTracksBranchName.c_str(), &tracksPtr);
  tree.GetBranch(mOutTPCITSTracksBranchName.c_str())->Fill();
  LOG(INFO) << "Writing " << tracks.size() << " TPC-ITS matches";
  if (mUseMC) {
    auto lblITS = pc.inputs().get<const std::vector<o2::MCCompLabel>>("matchITSMC");
    auto lblTPC = pc.inputs().get<const std::vector<o2::MCCompLabel>>("matchTPCMC");
    auto lblITSPtr = &lblITS;
    auto lblTPCPtr = &lblTPC;
    tree.Branch(mOutITSMCTruthBranchName.c_str(), &lblITSPtr);
    tree.Branch(mOutTPCMCTruthBranchName.c_str(), &lblTPCPtr);
    tree.GetBranch(mOutITSMCTruthBranchName.c_str())->Fill();
    tree.GetBranch(mOutTPCMCTruthBranchName.c_str())->Fill();
    //
  }
  tree.SetEntries(1);
  tree.Write();
  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(false);
}

DataProcessorSpec getTrackWriterTPCITSSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("match", "GLO", "TPCITS", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("matchITSMC", "GLO", "TPCITS_ITSMC", 0, Lifetime::Timeframe);
    inputs.emplace_back("matchTPCMC", "GLO", "TPCITS_TPCMC", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    "itstpc-track-writer",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<TrackWriterTPCITS>(useMC)},
    Options{
      {"tpcits-tracks-outfile", VariantType::String, "o2match_itstpc.root", {"Name of the output file"}}}};
}

} // namespace globaltracking
} // namespace o2
