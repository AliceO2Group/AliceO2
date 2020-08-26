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
#include "Analysis/MC.h"

#include <TH1F.h>
#include <cmath>

using namespace o2;
using namespace o2::framework;

struct QATrackingKine {
  OutputObj<TH1F> hpt{TH1F("pt", ";p_{T} [GeV]", 100, 0., 200.)};
  OutputObj<TH1F> hphi{TH1F("phi", ";#phi", 100, 0, 2 * M_PI)};
  OutputObj<TH1F> heta{TH1F("eta", ";#eta", 100, -6, 6)};

  void process(aod::Collision const& collision,
               aod::BCs const& bcs,
               soa::Join<aod::Tracks, aod::TracksCov> const& tracks)
  {
    for (auto& track : tracks) {
      heta->Fill(track.eta());
      hphi->Fill(track.phi());
      hpt->Fill(track.pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<QATrackingKine>("qa-tracking-kine"),
  };
}
