// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackWriterSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ControlService.h"
#include "ITSWorkflow/TrackWriterSpec.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace its
{

void TrackWriter::init(InitContext& ic)
{
  auto filename = ic.options().get<std::string>("its-track-outfile");
  mFile = std::make_unique<TFile>(filename.c_str(), "RECREATE");
  if (!mFile->IsOpen()) {
    LOG(ERROR) << "Cannot open the " << filename.c_str() << " file !";
    mState = 0;
    return;
  }
  mState = 1;
}

void TrackWriter::run(ProcessingContext& pc)
{
  if (mState != 1)
    return;

  auto tracks = pc.inputs().get<const std::vector<o2::its::TrackITS>>("tracks");
  auto clusIdx = pc.inputs().get<gsl::span<int>>("trackClIdx");
  auto rofs = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> labels;
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* plabels = nullptr;
  std::vector<int> clusIdxOut, *clusIdxOutPtr = &clusIdxOut;
  clusIdxOut.reserve(clusIdx.size());
  for (auto v : clusIdx) {
    clusIdxOut.push_back(v);
  }

  LOG(INFO) << "ITSTrackWriter pulled " << tracks.size() << " tracks, in "
            << rofs.size() << " RO frames";

  TTree tree("o2sim", "Tree with ITS tracks");
  tree.Branch("ITSTrack", &tracks);
  tree.Branch("ITSTrackClusIdx", &clusIdxOutPtr);
  if (mUseMC) {
    labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
    plabels = labels.get();
    tree.Branch("ITSTrackMCTruth", &plabels);
  }
  tree.Fill();
  tree.Write();

  // write ROFrecords vector to a tree
  TTree treeROF("ITSTracksROF", "ROF records tree");
  auto* rofsPtr = &rofs;
  treeROF.Branch("ITSTracksROF", &rofsPtr);
  treeROF.Fill();
  treeROF.Write();

  if (mUseMC) {
    // write MC2ROFrecord vector (directly inherited from digits input) to a tree
    TTree treeMC2ROF("ITSTracksMC2ROF", "MC -> ROF records tree");
    auto mc2rofs = pc.inputs().get<const std::vector<o2::itsmft::MC2ROFRecord>>("MC2ROframes");
    auto* mc2rofsPtr = &mc2rofs;
    treeMC2ROF.Branch("ITSTracksMC2ROF", &mc2rofsPtr);
    treeMC2ROF.Fill();
    treeMC2ROF.Write();
  }

  mFile->Close();

  mState = 2;
  pc.services().get<ControlService>().readyToQuit(false);
}

DataProcessorSpec getTrackWriterSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tracks", "ITS", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackClIdx", "ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("labels", "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "ITS", "ITSTrackMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-track-writer",
    inputs,
    Outputs{},
    AlgorithmSpec{ adaptFromTask<TrackWriter>(useMC) },
    Options{
      { "its-track-outfile", VariantType::String, "o2trac_its.root", { "Name of the output file" } } }
  };
}

} // namespace its
} // namespace o2
