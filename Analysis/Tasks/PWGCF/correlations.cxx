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

#include "Analysis/EventSelection.h"
#include "Analysis/TrackSelectionTables.h"
#include "Analysis/Centrality.h"
#include "Analysis/StepTHn.h"
#include "Analysis/CorrelationContainer.h"
#include "Analysis/PairCuts.h"

#include <TH1F.h>
#include <cmath>
#include <TDirectory.h>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

#define O2_DEFINE_CONFIGURABLE(NAME, TYPE, DEFAULT, HELP) Configurable<TYPE> NAME{#NAME, DEFAULT, HELP};

struct CorrelationTask {

  // Configuration
  O2_DEFINE_CONFIGURABLE(cfgCutVertex, float, 7.0f, "Accepted z-vertex range")
  O2_DEFINE_CONFIGURABLE(cfgCutPt, float, 0.5f, "Minimal pT for tracks")
  O2_DEFINE_CONFIGURABLE(cfgCutEta, float, 0.8f, "Eta range for tracks")

  O2_DEFINE_CONFIGURABLE(cfgPtOrder, int, 1, "Only consider pairs for which pT,1 < pT,2 (0 = OFF, 1 = ON)");
  O2_DEFINE_CONFIGURABLE(cfgTriggerCharge, int, 0, "Select on charge of trigger particle: 0 = all; 1 = positive; -1 = negative");
  O2_DEFINE_CONFIGURABLE(cfgAssociatedCharge, int, 0, "Select on charge of associated particle: 0 = all; 1 = positive; -1 = negative");
  O2_DEFINE_CONFIGURABLE(cfgPairCharge, int, 0, "Select on charge of particle pair: 0 = all; 1 = like sign; -1 = unlike sign");

  O2_DEFINE_CONFIGURABLE(cfgTwoTrackCut, float, -1, "Two track cut: -1 = off; >0 otherwise distance value (suggested: 0.02)");
  O2_DEFINE_CONFIGURABLE(cfgTwoTrackCutMinRadius, float, 0.8f, "Two track cut: radius in m from which two track cuts are applied");

  O2_DEFINE_CONFIGURABLE(cfgPairCutPhoton, float, -1, "Pair cut on photons: -1 = off; >0 otherwise distance value (suggested: 0.004)")
  O2_DEFINE_CONFIGURABLE(cfgPairCutK0, float, -1, "Pair cut on K0s: -1 = off; >0 otherwise distance value (suggested: 0.005)")
  O2_DEFINE_CONFIGURABLE(cfgPairCutLambda, float, -1, "Pair cut on Lambda: -1 = off; >0 otherwise distance value (suggested: 0.005)")
  O2_DEFINE_CONFIGURABLE(cfgPairCutPhi, float, -1, "Pair cut on Phi: -1 = off; >0 otherwise distance value")
  O2_DEFINE_CONFIGURABLE(cfgPairCutRho, float, -1, "Pair cut on Rho: -1 = off; >0 otherwise distance value")

  // Filters and input definitions
  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && (aod::track::pt > cfgCutPt) && ((aod::track::isGlobalTrack == true) || (aod::track::isGlobalTrackSDD == true));
  using myTracks = soa::Filtered<soa::Join<aod::Tracks, aod::TrackSelection>>;

  // Output definitions
  OutputObj<CorrelationContainer> same{"sameEvent"};
  OutputObj<CorrelationContainer> mixed{"mixedEvent"};
  //OutputObj<TDirectory> qaOutput{"qa"};

  struct Config {
    bool mPairCuts = false;
    //THn* mEfficiencyTrigger = nullptr;
    //THn* mEfficiencyAssociated = nullptr;
  } cfg;

  // HistogramRegistry registry{"qa", true, {
  //   {"yields", "centrality vs pT vs eta",  {HistogramType::kTH3F, { {100, 0, 100, "centrality"}, {40, 0, 20, "p_{T}"}, {100, -2, 2, "#eta"} }}},
  //   {"etaphi", "centrality vs eta vs phi", {HistogramType::kTH3F, { {100, 0, 100, "centrality"}, {100, -2, 2, "#eta"}, {200, 0, 2 * M_PI, "#varphi"} }}}
  // }};

  OutputObj<TH3F> yields{TH3F("yields", "centrality vs pT vs eta", 100, 0, 100, 40, 0, 20, 100, -2, 2)};
  OutputObj<TH3F> etaphi{TH3F("etaphi", "centrality vs eta vs phi", 100, 0, 100, 100, -2, 2, 200, 0, 2 * M_PI)};

  PairCuts mPairCuts;

  void init(o2::framework::InitContext&)
  {
    // --- CONFIGURATION ---
    const char* binning =
      "vertex: 7 | -7, 7\n"
      "delta_phi: 72 | -1.570796, 4.712389\n"
      "delta_eta: 40 | -2.0, 2.0\n"
      "p_t_assoc: 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\n"
      "p_t_trigger: 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0\n"
      "multiplicity: 0, 5, 10, 20, 30, 40, 50, 100.1\n"
      "eta: 20 | -1.0, 1.0\n"
      "p_t_leading: 100 | 0.0, 50.0\n"
      "p_t_leading_course: 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0\n"
      "p_t_eff: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0\n"
      "vertex_eff: 10 | -10, 10\n";

    if (cfgPairCutPhoton > 0 || cfgPairCutK0 > 0 || cfgPairCutLambda > 0 || cfgPairCutPhi > 0 || cfgPairCutRho > 0) {
      mPairCuts.SetPairCut(PairCuts::Photon, cfgPairCutPhoton);
      mPairCuts.SetPairCut(PairCuts::K0, cfgPairCutK0);
      mPairCuts.SetPairCut(PairCuts::Lambda, cfgPairCutLambda);
      mPairCuts.SetPairCut(PairCuts::Phi, cfgPairCutPhi);
      mPairCuts.SetPairCut(PairCuts::Rho, cfgPairCutRho);
      cfg.mPairCuts = true;
    }

    if (cfgTwoTrackCut > 0) {
      mPairCuts.SetTwoTrackCuts(cfgTwoTrackCut, cfgTwoTrackCutMinRadius);
    }

    // --- OBJECT INIT ---
    same.setObject(new CorrelationContainer("sameEvent", "sameEvent", "NumberDensityPhiCentralityVtx", binning));
    mixed.setObject(new CorrelationContainer("mixedEvent", "mixedEvent", "NumberDensityPhiCentralityVtx", binning));
    //qaOutput.setObject(new TDirectory("qa", "qa"));
  }

  // Version with explicit nested loop
  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::Cents>>::iterator const& collision, myTracks const& tracks)
  {
    LOGF(info, "Tracks for collision: %d | Vertex: %.1f | INT7: %d | V0M: %.1f", tracks.size(), collision.posZ(), collision.sel7(), collision.centV0M());

    const auto centrality = collision.centV0M();

    same->fillEvent(centrality, CorrelationContainer::kCFStepAll);

    if (!collision.sel7())
      return;

    same->fillEvent(centrality, CorrelationContainer::kCFStepTriggered);

    // vertex already checked as filter
    same->fillEvent(centrality, CorrelationContainer::kCFStepVertex);

    same->fillEvent(centrality, CorrelationContainer::kCFStepReconstructed);

    int bSign = 1; // TODO magnetic field from CCDB

    for (auto& track1 : tracks) {

      // LOGF(info, "Track %f | %f | %f  %d %d", track1.eta(), track1.phi(), track1.pt(), track1.isGlobalTrack(), track1.isGlobalTrackSDD());

      // control histograms
      // ((TH3*) (registry.get("yields").get()))->Fill(centrality, track1.pt(), track1.eta());
      // ((TH3*) (registry.get("etaphi").get()))->Fill(centrality, track1.eta(), track1.phi());
      yields->Fill(centrality, track1.pt(), track1.eta());
      etaphi->Fill(centrality, track1.eta(), track1.phi());

      if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.charge() < 0)
        continue;

      double eventValues[3];
      eventValues[0] = track1.pt();
      eventValues[1] = centrality;
      eventValues[2] = collision.posZ();

      same->getTriggerHist()->Fill(eventValues, CorrelationContainer::kCFStepReconstructed);
      //mixed->getTriggerHist()->Fill(eventValues, CorrelationContainer::kCFStepReconstructed);

      for (auto& track2 : tracks) {
        if (track1 == track2)
          continue;

        if (cfgPtOrder != 0 && track2.pt() >= track1.pt())
          continue;

        if (cfgAssociatedCharge != 0 && cfgAssociatedCharge * track2.charge() < 0)
          continue;
        if (cfgPairCharge != 0 && cfgPairCharge * track1.charge() * track2.charge() < 0)
          continue;

        if (cfg.mPairCuts && mPairCuts.conversionCuts(track1, track2))
          continue;

        if (cfgTwoTrackCut > 0 && mPairCuts.twoTrackCut(track1, track2, bSign))
          continue;

        double values[6] = {0};

        values[0] = track1.eta() - track2.eta();
        values[1] = track2.pt();
        values[2] = track1.pt();
        values[3] = centrality;

        values[4] = track1.phi() - track2.phi();
        if (values[4] > 1.5 * TMath::Pi())
          values[4] -= TMath::TwoPi();
        if (values[4] < -0.5 * TMath::Pi())
          values[4] += TMath::TwoPi();

        values[5] = collision.posZ();

        same->getPairHist()->Fill(values, CorrelationContainer::kCFStepReconstructed);
        //mixed->getPairHist()->Fill(values, CorrelationContainer::kCFStepReconstructed);
      }
    }
  }

  // Version with combinations
  void process2(aod::Collision const& collision, soa::Filtered<aod::Tracks> const& tracks)
  {
    LOGF(info, "Tracks for collision (Combination run): %d", tracks.size());

    int bSign = 1; // TODO magnetic field from CCDB

    for (auto track1 = tracks.begin(); track1 != tracks.end(); ++track1) {

      if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.charge() < 0)
        continue;

      //       LOGF(info, "TRACK %f %f | %f %f | %f %f", track1.eta(), track1.eta(), track1.phi(), track1.phi2(), track1.pt(), track1.pt());

      double eventValues[3];
      eventValues[0] = track1.pt();
      eventValues[1] = 0; // collision.v0mult();
      eventValues[2] = collision.posZ();

      same->getTriggerHist()->Fill(eventValues, CorrelationContainer::kCFStepReconstructed);
      //mixed->getTriggerHist()->Fill(eventValues, CorrelationContainer::kCFStepReconstructed);
    }

    for (auto& [track1, track2] : combinations(tracks, tracks)) {
      //LOGF(info, "Combination %d %d", track1.index(), track2.index());

      if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.charge() < 0)
        continue;
      if (cfgAssociatedCharge != 0 && cfgAssociatedCharge * track2.charge() < 0)
        continue;
      if (cfgPairCharge != 0 && cfgPairCharge * track1.charge() * track2.charge() < 0)
        continue;

      if (cfg.mPairCuts && mPairCuts.conversionCuts(track1, track2))
        continue;

      if (cfgTwoTrackCut > 0 && mPairCuts.twoTrackCut(track1, track2, bSign))
        continue;

      double values[6] = {0};

      values[0] = track1.eta() - track2.eta();
      values[1] = track1.pt();
      values[2] = track2.pt();
      values[3] = 0; // collision.v0mult();

      values[4] = track1.phi() - track2.phi();
      if (values[4] > 1.5 * TMath::Pi())
        values[4] -= TMath::TwoPi();
      if (values[4] < -0.5 * TMath::Pi())
        values[4] += TMath::TwoPi();

      values[5] = collision.posZ();

      same->getPairHist()->Fill(values, CorrelationContainer::kCFStepReconstructed);
      //mixed->getPairHist()->Fill(values, CorrelationContainer::kCFStepReconstructed);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<CorrelationTask>("correlation-task")};
}
