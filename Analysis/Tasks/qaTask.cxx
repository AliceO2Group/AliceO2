// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Peter Hristov <Peter.Hristov@cern.ch>, CERN
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Analysis/MC.h"

#include <TH1F.h>
#include <TH2F.h>
#include <cmath>

using namespace o2;
using namespace o2::framework;

struct QATrackingKine {
  Configurable<bool> ismc{"ismc", false, "option to flag mc"};
  OutputObj<TH1F> hpt{TH1F("pt", ";p_{T} [GeV]", 100, 0., 10.)};
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

struct QATrackingResolution {
  OutputObj<TH1F> etaDiff{TH1F("etaDiff", ";eta_{MC} - eta_{Rec}", 100, -2, 2)};
  OutputObj<TH1F> phiDiff{TH1F("phiDiff", ";phi_{MC} - phi_{Rec}", 100, -M_PI, M_PI)};
  OutputObj<TH1F> ptDiff{TH1F("ptDiff", ";p_{T}_{MC} - p_{T}_{Rec}", 400, -2., 2.)};
  OutputObj<TH1F> ptRes{TH1F("ptDRes", ";p_{T}_{MC} - p_{T}_{Rec} / p_{T}_{Rec} ", 400, -2., 2.)};
  OutputObj<TH2F> ptResvspt{TH2F("ptDResvspt", ";p_{T};Res p_{T}", 100, 0., 10., 400, -2., 2.)};
  OutputObj<TH2F> ptResvseta{TH2F("ptDResvseta", ";#eta;Res p_{T}", 400, -4., 4., 400, -2., 2.)};

  void process(soa::Join<aod::Collisions, aod::McCollisionLabels>::iterator const& collision, soa::Join<aod::Tracks, aod::McTrackLabels> const& tracks, aod::McParticles const& mcParticles, aod::McCollisions const& mcCollisions)
  {
    LOGF(info, "vtx-z (data) = %f | vtx-z (MC) = %f", collision.posZ(), collision.label().posZ());
    for (auto& track : tracks) {
      ptDiff->Fill(track.label().pt() - track.pt());
      ptRes->Fill((track.label().pt() - track.pt()) / track.pt());
      ptResvspt->Fill(track.pt(), abs(track.label().pt() - track.pt()) / track.pt());
      ptResvseta->Fill(track.eta(), abs(track.label().pt() - track.pt()) / track.pt());
      etaDiff->Fill(track.label().eta() - track.eta());

      auto delta = track.label().phi() - track.phi();
      if (delta > M_PI) {
        delta -= 2 * M_PI;
      }
      if (delta < -M_PI) {
        delta += 2 * M_PI;
      }
      phiDiff->Fill(delta);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<QATrackingKine>("qa-tracking-kine"),
    adaptAnalysisTask<QATrackingResolution>("qa-tracking-resolution"),
  };
}
