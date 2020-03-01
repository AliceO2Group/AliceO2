// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Analysis/SecondaryVertex.h"
#include "DetectorsBase/DCAFitter.h"
#include "ReconstructionDataFormats/Track.h"

#include <TFile.h>
#include <TH1F.h>
#include <cmath>
#include <array>

struct TrackKineTask {
  OutputObj<TH1F> hvtx_x_out{TH1F("hvtx_x", "2-track vtx", 100, -0.1, 0.1)};
  void process(aod::Collision const& collision, aod::TracksMCtruth const& trackskine)
  {
    //LOGF(info, "testing");
    LOGF(info, "Tracks for collision: %d", trackskine.size());
    //for (auto it_0 = trackskine.begin(); it_0 != trackskine.end(); ++it_0) {
    //  auto& trackkine_0 = *it_0;
      //LOGF(info, "look at PDG code here %d", trackkine_0.pdgCode());
    //}
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<TrackKineTask>("trackkine-trackkinetask")};
}
