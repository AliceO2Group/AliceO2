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

#include "MFTWorkflow/TrackWriterSpec.h"
#include "MFTTracking/TrackCA.h"

#include "Framework/ControlService.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace mft
{

void TrackWriter::init(InitContext& ic)
{
  auto filename = ic.options().get<std::string>("mft-track-outfile");
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

  auto tracks = pc.inputs().get<const std::vector<o2::mft::TrackMFT>>("tracks");
  auto tracksltf = pc.inputs().get<const std::vector<o2::mft::TrackLTF>>("tracksltf");
  auto tracksca = pc.inputs().get<const std::vector<o2::mft::TrackCA>>("tracksca");
  auto rofs = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> labels;
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* plabels = nullptr;

  LOG(INFO) << "MFTTrackWriter pulled "
            << tracks.size() << " tracks, "
            << tracksltf.size() << " tracks LTF, "
            << tracksca.size() << " tracks CA, in "
            << rofs.size() << " RO frames";

  TTree tree("o2sim", "Tree with MFT tracks");
  tree.Branch("MFTTrack", &tracks);
  tree.Branch("MFTTrackLTF", &tracksltf);
  tree.Branch("MFTTrackCA", &tracksca);
  if (mUseMC) {
    labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
    plabels = labels.get();
    tree.Branch("MFTTrackMCTruth", &plabels);
  }
  tree.Fill();
  tree.Write();

  // write ROFrecords vector to a tree
  TTree treeROF("MFTTracksROF", "ROF records tree");
  auto* rofsPtr = &rofs;
  treeROF.Branch("MFTTracksROF", &rofsPtr);
  treeROF.Fill();
  treeROF.Write();

  if (mUseMC) {
    // write MC2ROFrecord vector (directly inherited from digits input) to a tree
    TTree treeMC2ROF("MFTTracksMC2ROF", "MC -> ROF records tree");
    auto mc2rofs = pc.inputs().get<const std::vector<o2::itsmft::MC2ROFRecord>>("MC2ROframes");
    auto* mc2rofsPtr = &mc2rofs;
    treeMC2ROF.Branch("MFTTracksMC2ROF", &mc2rofsPtr);
    treeMC2ROF.Fill();
    treeMC2ROF.Write();
  }

  mFile->Close();

  mState = 2;
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getTrackWriterSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tracks", "MFT", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracksltf", "MFT", "TRACKSLTF", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracksca", "MFT", "TRACKSCA", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "MFT", "MFTTrackROF", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("labels", "MFT", "TRACKSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "MFT", "MFTTrackMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "mft-track-writer",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<TrackWriter>(useMC)},
    Options{
      {"mft-track-outfile", VariantType::String, "mfttracks.root", {"Name of the output file"}}}};
}

} // namespace mft
} // namespace o2
