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
#include <CCDB/BasicCCDBManager.h>
#include "Framework/StepTHn.h"
#include "Framework/HistogramRegistry.h"

#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/TrackSelectionTables.h"
#include "AnalysisDataModel/Centrality.h"
#include "AnalysisCore/CorrelationContainer.h"
#include "AnalysisCore/PairCuts.h"

#include <TH1F.h>
#include <cmath>
#include <TDirectory.h>
#include <THn.h>

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

  O2_DEFINE_CONFIGURABLE(cfgEfficiencyTrigger, std::string, "", "CCDB path to efficiency object for trigger particles")
  O2_DEFINE_CONFIGURABLE(cfgEfficiencyAssociated, std::string, "", "CCDB path to efficiency object for associated particles")

  // Filters and input definitions
  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;
  Filter trackFilter = (nabs(aod::track::eta) < cfgCutEta) && (aod::track::pt > cfgCutPt) && ((aod::track::isGlobalTrack == (uint8_t) true) || (aod::track::isGlobalTrackSDD == (uint8_t) true));
  using myTracks = soa::Filtered<soa::Join<aod::Tracks, aod::TrackSelection>>;

  // Output definitions
  OutputObj<CorrelationContainer> same{"sameEvent"};
  OutputObj<CorrelationContainer> mixed{"mixedEvent"};

  struct Config {
    bool mPairCuts = false;
    THn* mEfficiencyTrigger = nullptr;
    THn* mEfficiencyAssociated = nullptr;
  } cfg;

  HistogramRegistry registry{"registry", {
                                           {"yields", "centrality vs pT vs eta", {HistType::kTH3F, {{100, 0, 100, "centrality"}, {40, 0, 20, "p_{T}"}, {100, -2, 2, "#eta"}}}},          //
                                           {"etaphi", "centrality vs eta vs phi", {HistType::kTH3F, {{100, 0, 100, "centrality"}, {100, -2, 2, "#eta"}, {200, 0, 2 * M_PI, "#varphi"}}}} //
                                         }};

  PairCuts mPairCuts;

  Service<o2::ccdb::BasicCCDBManager> ccdb;

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

    mPairCuts.SetHistogramRegistry(&registry);

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

    // o2-ccdb-upload -p Users/jgrosseo/correlations/LHC15o -f /tmp/correction_2011_global.root -k correction

    ccdb->setURL("http://ccdb-test.cern.ch:8080");
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();

    long now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    ccdb->setCreatedNotAfter(now); // TODO must become global parameter from the train creation time

    if (cfgEfficiencyTrigger.value.empty() == false) {
      cfg.mEfficiencyTrigger = ccdb->getForTimeStamp<THnT<float>>(cfgEfficiencyTrigger, now);
      LOGF(info, "Loaded efficiency histogram for trigger particles from %s (%p)", cfgEfficiencyTrigger.value.c_str(), (void*)cfg.mEfficiencyTrigger);
    }
    if (cfgEfficiencyAssociated.value.empty() == false) {
      cfg.mEfficiencyAssociated = ccdb->getForTimeStamp<THnT<float>>(cfgEfficiencyAssociated, now);
      LOGF(info, "Loaded efficiency histogram for associated particles from %s (%p)", cfgEfficiencyAssociated.value.c_str(), (void*)cfg.mEfficiencyAssociated);
    }
  }

  // Version with explicit nested loop
  void process(soa::Filtered<soa::Join<aod::Collisions, aod::EvSels, aod::Cents>>::iterator const& collision, aod::BCsWithTimestamps const&, myTracks const& tracks)
  {
    auto bc = collision.bc_as<aod::BCsWithTimestamps>();
    if (cfgEfficiencyTrigger.value.empty() == false) {
      cfg.mEfficiencyTrigger = ccdb->getForTimeStamp<THnT<float>>(cfgEfficiencyTrigger, bc.timestamp());
      LOGF(info, "Loaded efficiency histogram for trigger particles from %s (%p)", cfgEfficiencyTrigger.value.c_str(), (void*)cfg.mEfficiencyTrigger);
    }
    if (cfgEfficiencyAssociated.value.empty() == false) {
      cfg.mEfficiencyAssociated = ccdb->getForTimeStamp<THnT<float>>(cfgEfficiencyAssociated, bc.timestamp());
      LOGF(info, "Loaded efficiency histogram for associated particles from %s (%p)", cfgEfficiencyAssociated.value.c_str(), (void*)cfg.mEfficiencyAssociated);
    }

    LOGF(info, "Tracks for collision: %d | Vertex: %.1f | INT7: %d | V0M: %.1f", tracks.size(), collision.posZ(), collision.sel7(), collision.centV0M());

    const auto centrality = collision.centV0M();

    same->fillEvent(centrality, CorrelationContainer::kCFStepAll);

    if (!collision.alias()[kINT7]) {
      return;
    }
    if (!collision.sel7()) {
      return;
    }

    same->fillEvent(centrality, CorrelationContainer::kCFStepTriggered);

    // vertex already checked as filter
    same->fillEvent(centrality, CorrelationContainer::kCFStepVertex);

    same->fillEvent(centrality, CorrelationContainer::kCFStepReconstructed);

    int bSign = 1; // TODO magnetic field from CCDB

    // Cache efficiency for particles (too many FindBin lookups)
    float* efficiencyAssociated = nullptr;
    if (cfg.mEfficiencyAssociated) {
      efficiencyAssociated = new float[tracks.size()];
      int i = 0;
      for (auto& track1 : tracks) {
        efficiencyAssociated[i++] = getEfficiency(cfg.mEfficiencyAssociated, track1.eta(), track1.pt(), centrality, collision.posZ());
      }
    }

    for (auto& track1 : tracks) {
      // LOGF(info, "Track %f | %f | %f  %d %d", track1.eta(), track1.phi(), track1.pt(), track1.isGlobalTrack(), track1.isGlobalTrackSDD());

      registry.fill(HIST("yields"), centrality, track1.pt(), track1.eta());
      registry.fill(HIST("etaphi"), centrality, track1.eta(), track1.phi());

      if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.charge() < 0) {
        continue;
      }

      float triggerWeight = 1.0;
      if (cfg.mEfficiencyTrigger) {
        triggerWeight = getEfficiency(cfg.mEfficiencyTrigger, track1.eta(), track1.pt(), centrality, collision.posZ());
      }

      same->getTriggerHist()->Fill(CorrelationContainer::kCFStepReconstructed, track1.pt(), centrality, collision.posZ(), triggerWeight);
      //mixed->getTriggerHist()->Fill(eventValues, CorrelationContainer::kCFStepReconstructed);

      int i = -1;
      for (auto& track2 : tracks) {
        i++; // HACK
        if (track1 == track2) {
          continue;
        }

        if (cfgPtOrder != 0 && track2.pt() >= track1.pt()) {
          continue;
        }

        if (cfgAssociatedCharge != 0 && cfgAssociatedCharge * track2.charge() < 0) {
          continue;
        }
        if (cfgPairCharge != 0 && cfgPairCharge * track1.charge() * track2.charge() < 0) {
          continue;
        }

        if (cfg.mPairCuts && mPairCuts.conversionCuts(track1, track2)) {
          continue;
        }

        if (cfgTwoTrackCut > 0 && mPairCuts.twoTrackCut(track1, track2, bSign)) {
          continue;
        }

        float associatedWeight = 1.0;
        if (cfg.mEfficiencyAssociated) {
          associatedWeight = efficiencyAssociated[i];
        }

        float deltaPhi = track1.phi() - track2.phi();
        if (deltaPhi > 1.5 * TMath::Pi()) {
          deltaPhi -= TMath::TwoPi();
        }
        if (deltaPhi < -0.5 * TMath::Pi()) {
          deltaPhi += TMath::TwoPi();
        }

        same->getPairHist()->Fill(CorrelationContainer::kCFStepReconstructed,
                                  track1.eta() - track2.eta(), track2.pt(), track1.pt(), centrality, deltaPhi, collision.posZ(),
                                  triggerWeight * associatedWeight);
        //mixed->getPairHist()->Fill(values, CorrelationContainer::kCFStepReconstructed);
      }
    }

    delete[] efficiencyAssociated;
  }

  // Version with combinations
  void process2(soa::Join<aod::Collisions, aod::Cents>::iterator const& collision, soa::Filtered<aod::Tracks> const& tracks)
  {
    LOGF(info, "Tracks for collision (Combination run): %d", tracks.size());

    const auto centrality = collision.centV0M();

    int bSign = 1; // TODO magnetic field from CCDB

    for (auto track1 = tracks.begin(); track1 != tracks.end(); ++track1) {

      if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.charge() < 0) {
        continue;
      }

      //       LOGF(info, "TRACK %f %f | %f %f | %f %f", track1.eta(), track1.eta(), track1.phi(), track1.phi2(), track1.pt(), track1.pt());

      same->getTriggerHist()->Fill(CorrelationContainer::kCFStepReconstructed, track1.pt(), centrality, collision.posZ());
      //mixed->getTriggerHist()->Fill(eventValues, CorrelationContainer::kCFStepReconstructed);
    }

    for (auto& [track1, track2] : combinations(tracks, tracks)) {
      //LOGF(info, "Combination %d %d", track1.index(), track2.index());

      if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.charge() < 0) {
        continue;
      }
      if (cfgAssociatedCharge != 0 && cfgAssociatedCharge * track2.charge() < 0) {
        continue;
      }
      if (cfgPairCharge != 0 && cfgPairCharge * track1.charge() * track2.charge() < 0) {
        continue;
      }

      if (cfg.mPairCuts && mPairCuts.conversionCuts(track1, track2)) {
        continue;
      }

      if (cfgTwoTrackCut > 0 && mPairCuts.twoTrackCut(track1, track2, bSign)) {
        continue;
      }

      float deltaPhi = track1.phi() - track2.phi();
      if (deltaPhi > 1.5 * TMath::Pi()) {
        deltaPhi -= TMath::TwoPi();
      }
      if (deltaPhi < -0.5 * TMath::Pi()) {
        deltaPhi += TMath::TwoPi();
      }

      same->getPairHist()->Fill(CorrelationContainer::kCFStepReconstructed,
                                track1.eta() - track2.eta(), track2.pt(), track1.pt(), centrality, deltaPhi, collision.posZ());
      //mixed->getPairHist()->Fill(values, CorrelationContainer::kCFStepReconstructed);
    }
  }

  double getEfficiency(THn* eff, float eta, float pt, float centrality, float posZ)
  {
    int effVars[4];
    effVars[0] = eff->GetAxis(0)->FindBin(eta);
    effVars[1] = eff->GetAxis(1)->FindBin(pt);
    effVars[2] = eff->GetAxis(2)->FindBin(centrality);
    effVars[3] = eff->GetAxis(3)->FindBin(posZ);
    return eff->GetBinContent(effVars);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<CorrelationTask>("correlation-task")};
}
