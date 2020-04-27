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
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "GlobalTrackingWorkflow/TrackWriterTPCITSSpec.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonUtils/StringUtils.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

template <typename T>
TBranch* getOrMakeBranch(TTree* tree, const char* brname, T* ptr)
{
  if (auto br = tree->GetBranch(brname)) {
    br->SetAddress(static_cast<void*>(ptr));
    return br;
  }
  return tree->Branch(brname, ptr); // otherwise make it
}

void TrackWriterTPCITS::init(InitContext& ic)
{
  mOutFileName = ic.options().get<std::string>("tpcits-tracks-outfile");
  mFile = std::make_unique<TFile>(mOutFileName.c_str(), "RECREATE");
  if (!mFile->IsOpen()) {
    throw std::runtime_error(o2::utils::concat_string("failed to open TPC-ITS matches output file ", mOutFileName));
  }
  mTree = std::make_unique<TTree>(mTreeName.c_str(), "Tree of ITS-TPC matches");
}

void TrackWriterTPCITS::run(ProcessingContext& pc)
{

  auto tracks = std::move(pc.inputs().get<const std::vector<o2::dataformats::TrackTPCITS>>("match"));
  auto tracksPtr = &tracks;
  getOrMakeBranch(mTree.get(), mOutTPCITSTracksBranchName.c_str(), &tracksPtr);
  std::vector<o2::MCCompLabel> lblITS, *lblITSPtr = &lblITS;
  std::vector<o2::MCCompLabel> lblTPC, *lblTPCPtr = &lblTPC;
  if (mUseMC) {
    lblITS = std::move(pc.inputs().get<std::vector<o2::MCCompLabel>>("matchITSMC"));
    lblTPC = std::move(pc.inputs().get<std::vector<o2::MCCompLabel>>("matchTPCMC"));
    getOrMakeBranch(mTree.get(), mOutITSMCTruthBranchName.c_str(), &lblITSPtr);
    getOrMakeBranch(mTree.get(), mOutTPCMCTruthBranchName.c_str(), &lblTPCPtr);
  }
  LOG(INFO) << "Writing " << tracks.size() << " TPC-ITS matches";
  mTree->Fill();
}

void TrackWriterTPCITS::endOfStream(EndOfStreamContext& ec)
{
  LOG(INFO) << "Finalizing TPC-ITS matched tracks writing";
  mTree->Write();
  mTree.release()->Delete();
  mFile->Close();
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
