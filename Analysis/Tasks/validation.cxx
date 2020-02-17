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
#include "DetectorsBase/DCAFitter.h"
#include "ReconstructionDataFormats/Track.h"

#include <TFile.h>
#include <TH1F.h>
#include <cmath>
#include <array>
namespace o2::aod
{
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

struct ValidationTask {
  OutputObj<TH1F> hpt_nocuts{TH1F("hpt_nocuts", "pt tracks (#GeV)", 100, 0., 10.)};
  OutputObj<TH1F> hrun_number{TH1F("hrun_number", "run number", 1000, 0., 1000000.)};

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksCov> const& tracks)
  {
    hrun_number->Fill(collision.runNumber());
    LOGF(info, "Tracks for collision: %d", tracks.size());
    for (auto& track : tracks) {
      hpt_nocuts->Fill(track.pt());
      LOGF(info, "track tgl  ss   s%f", track.tgl());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ValidationTask>("validation-qa")};
}
