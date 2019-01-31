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
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace ITS
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

  auto tracks = pc.inputs().get<const std::vector<o2::ITS::TrackITS>>("tracks");
  auto labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
  auto plabels = labels.get();

  LOG(INFO) << "ITSTrackWriter pulled " << tracks.size() << " tracks, "
            << labels->getIndexedSize() << " MC label objects";

  TTree tree("o2sim", "Tree with ITS tracks");
  tree.Branch("ITSTrack", &tracks);
  tree.Branch("ITSTrackMCTruth", &plabels);
  tree.Fill();
  tree.Write();
  mFile->Close();

  mState = 2;
  pc.services().get<ControlService>().readyToQuit(true);
}

DataProcessorSpec getTrackWriterSpec()
{
  return DataProcessorSpec{
    "its-track-writer",
    Inputs{
      InputSpec{ "tracks", "ITS", "TRACKS", 0, Lifetime::Timeframe },
      InputSpec{ "labels", "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe } },
    Outputs{},
    AlgorithmSpec{ adaptFromTask<TrackWriter>() },
    Options{
      { "its-track-outfile", VariantType::String, "o2trac_its.root", { "Name of the output file" } } }
  };
}

} // namespace ITS
} // namespace o2
