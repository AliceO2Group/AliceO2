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

#include "TFile.h"
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

DataProcessorSpec getTrackWriterSpec()
{
  auto init = [](InitContext& ic) {
    auto filename = ic.options().get<std::string>("its-track-outfile");

    return [filename](ProcessingContext& pc) {
      static bool done = false;
      if (done)
        return;

      TFile file(filename.c_str(), "RECREATE");
      if (file.IsOpen()) {
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
        file.Close();

      } else {
        LOG(ERROR) << "Cannot open the " << filename.c_str() << " file !";
      }
      done = true;
      pc.services().get<ControlService>().readyToQuit(true);
    };
  };

  return DataProcessorSpec{
    "its-track-writer",
    Inputs{
      InputSpec{ "tracks", "ITS", "TRACKS", 0, Lifetime::Timeframe },
      InputSpec{ "labels", "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe } },
    Outputs{},
    AlgorithmSpec{ init },
    Options{
      { "its-track-outfile", VariantType::String, "o2trac_its.root", { "Name of the output file" } } }
  };
}

} // namespace ITS
} // namespace o2
