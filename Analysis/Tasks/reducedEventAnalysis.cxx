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
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/Multiplicity.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Centrality.h"
#include <TH1F.h>
#include <TH2F.h>
#include <TMath.h>
#include "TVector3.h"
#include "TLorentzVector.h"
#include <cmath>
#include <vector>

const float gkMass = 0.0005;

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

// This is a very simple example showing how to create an histogram
// FIXME: this should really inherit from AnalysisTask but
//        we need GCC 7.4+ for that

struct InvMassAnalysis {
  // needs to be initialized with a label or an obj
  // when adding an object to OutputObj later, the object name will be
  // *reset* to OutputObj label - needed for correct placement in the output file
  OutputObj<TH1F> centV0M{TH1F("centV0M", "centrality V0", 100, 0.0, 100.0)};
  OutputObj<TH1F> vtxZ{TH1F("vtxZ", "vtx Z", 200, -20.0, 20.0)};

  OutputObj<TH1F> ptH{TH1F("pt", "pt", 100, -0.01, 10.01)};
  OutputObj<TH2F> ptCorr{TH2F("ptToPt", "ptToPt", 100, -0.01, 10.01, 100, -0.01, 10.01)};
  OutputObj<TH2F> tpcDedx{TH2F("tpcDedx", "TPC de/dx", 100, 0.0, 10.0, 100, 0.0, 200.0)};
  OutputObj<TH1I> itsHits{TH1I("itsHits", "ITS hits per layer", 6, -0.5, 5.5)};
  OutputObj<TH2I> itsHitsVsPt{TH2I("itsHitsVsPt", "ITS hits per layer", 6, -0.5, 5.5, 100, 0.0, 10.0)};
  OutputObj<TH1I> itsChi2{TH1I("itsChi2", "ITS chi2", 100, 0.0, 20.0)};
  OutputObj<TH1I> tpcChi2{TH1I("tpcChi2", "TPC chi2", 100, 0.0, 10.0)};
  OutputObj<TH1I> tpcCls{TH1I("tpcCls", "TPC clusters", 160, 0.0, 160.0)};
  OutputObj<TH1I> flagsHist{TH1I("flagsHist", "Flags", 64, -0.5, 63.5)};
  OutputObj<TH1F> invMassPM{TH1F("invMassPM", "Invariant mass, SEPM", 125, 0.0, 5.0)};
  OutputObj<TH1F> invMassPP{TH1F("invMassPP", "Invariant mass, SEPP", 125, 0.0, 5.0)};
  OutputObj<TH1F> invMassMM{TH1F("invMassMM", "Invariant mass, SEMM", 125, 0.0, 5.0)};
  OutputObj<TH2F> invMassVsPt{TH2F("invMassVsPt", "Invariant mass", 125, 0.0, 5.0, 10, 0.0, 10.0)};
  OutputObj<TH2F> invMassVsCentrality{TH2F("invMassVsCentrality", "Invariant mass", 125, 0.0, 5.0, 10, 0.0, 100.0)};
  OutputObj<TH1F> trZ{"trZ", OutputObjHandlingPolicy::QAObject};
  //Configurable<float> ptlow{"ptlow", 1.0f, "Lower pT limit"};
  //Configurable<float> pthigh{"pthigh", 1.0f, "Higher pT limit"};

  float ptlow = 1.0;
  float pthigh = 5.0;
  Filter ptFilter = ((1.0f / aod::track::signed1Pt > ptlow) && (1.0f / aod::track::signed1Pt < pthigh)) || ((1.0f / aod::track::signed1Pt > -1.0f * pthigh) && (1.0f / aod::track::signed1Pt < -1.0f * ptlow));
  //Filter spdAnyFilter = (aod::track::itsClusterMap & (uint8_t(1)<<0)) || (aod::track::itsClusterMap & (uint8_t(1)<<1));
  float dedxLow = 75.0;
  float dedxHigh = 90.0;
  Filter dedxFilter = (aod::track::tpcSignal > dedxLow) && (aod::track::tpcSignal < dedxHigh);
  float tpcChi2Max = 4.0;
  float itsChi2Max = 36;
  Filter qualityFilter = (aod::track::tpcChi2NCl < tpcChi2Max) && (aod::track::itsChi2NCl < itsChi2Max);

  void init(InitContext const&)
  {
    trZ.setObject(new TH1F("Z", "Z", 100, -10., 10.));
  }

  void process(soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator collision, soa::Filtered<soa::Join<aod::Tracks, aod::TracksExtra>> const& tracks)
  {

    if (!collision.sel7())
      return;

    centV0M->Fill(collision.centV0M());
    vtxZ->Fill(collision.posZ());

    for (auto& track : tracks) {
      //if (track.pt() < ptlow)
      // continue;
      ptH->Fill(track.pt());
      trZ->Fill(track.z());
      itsChi2->Fill(track.itsChi2NCl());
      for (int i = 0; i < 6; i++) {
        if (track.itsClusterMap() & (uint8_t(1) << i))
          itsHits->Fill(i);
        if (track.itsClusterMap() & (uint8_t(1) << i))
          itsHitsVsPt->Fill(i, track.pt());
      }
      tpcDedx->Fill(track.tpcInnerParam(), track.tpcSignal());
      tpcChi2->Fill(track.tpcChi2NCl());
      tpcCls->Fill(track.tpcNClsFound());
      for (int i = 0; i < 64; i++) {
        if (track.flags() & (uint64_t(1) << i))
          flagsHist->Fill(i);
      }
    }
    for (auto& [t0, t1] : combinations(tracks, tracks)) {
      ptCorr->Fill(t0.pt(), t1.pt());
      if (!((t0.itsClusterMap() & (uint8_t(1) << 0)) || (t0.itsClusterMap() & (uint8_t(1) << 1))))
        continue;
      if (!((t1.itsClusterMap() & (uint8_t(1) << 0)) || (t1.itsClusterMap() & (uint8_t(1) << 1))))
        continue;

      TLorentzVector p1, p2, p;
      p1.SetXYZM(t0.px(), t0.py(), t0.pz(), gkMass);
      p2.SetXYZM(t1.px(), t1.py(), t1.pz(), gkMass);
      p = p1 + p2;

      if (t0.charge() * t1.charge() < 0) {
        invMassPM->Fill(p.M());
        invMassVsPt->Fill(p.M(), p.Pt());
        invMassVsCentrality->Fill(p.M(), collision.centV0M());
      } else {
        if (t0.charge() > 0)
          invMassPP->Fill(p.M());
        if (t0.charge() < 0)
          invMassMM->Fill(p.M());
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<InvMassAnalysis>("InvMassAnalysis")};
}
