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
#include "Framework/StepTHn.h"

#include "Analysis/EventSelection.h"
#include "Analysis/TrackSelectionTables.h"
#include "Analysis/Centrality.h"
#include "Analysis/CorrelationContainer.h"
#include "Analysis/CFDerived.h"

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

  O2_DEFINE_CONFIGURABLE(cfgTwoTrackCut, float, -1, "Two track cut: -1 = off; >0 otherwise distance value");
  O2_DEFINE_CONFIGURABLE(cfgTwoTrackCutMinRadius, float, 0.8f, "Two track cut: radius in m from which two track cuts are applied");

  O2_DEFINE_CONFIGURABLE(cfgPairCutPhoton, float, -1, "Pair cut on photons: -1 = off; >0 otherwise distance value (suggested: 0.004)")
  O2_DEFINE_CONFIGURABLE(cfgPairCutK0, float, -1, "Pair cut on K0s: -1 = off; >0 otherwise distance value (suggested: 0.005)")
  O2_DEFINE_CONFIGURABLE(cfgPairCutLambda, float, -1, "Pair cut on Lambda: -1 = off; >0 otherwise distance value (suggested: 0.005)")
  O2_DEFINE_CONFIGURABLE(cfgPairCutPhi, float, -1, "Pair cut on Phi: -1 = off; >0 otherwise distance value")
  O2_DEFINE_CONFIGURABLE(cfgPairCutRho, float, -1, "Pair cut on Rho: -1 = off; >0 otherwise distance value")

  // Filters and input definitions
  Filter collisionFilter = nabs(aod::collision::posZ) < cfgCutVertex;
  Filter trackFilter = (nabs(aod::cftrack::eta) < cfgCutEta) && (aod::cftrack::pt > cfgCutPt);

  // Output definitions
  OutputObj<CorrelationContainer> same{"sameEvent"};
  OutputObj<CorrelationContainer> mixed{"mixedEvent"};
  //OutputObj<TDirectory> qaOutput{"qa"};

  enum PairCuts { Photon = 0,
                  K0,
                  Lambda,
                  Phi,
                  Rho };
  struct Config {
    bool mPairCuts = false;
    //THn* mEfficiencyTrigger = nullptr;
    //THn* mEfficiencyAssociated = nullptr;
  } cfg;

  struct QA {
    TH3F* mTwoTrackDistancePt[2] = {nullptr}; // control histograms for two-track efficiency study: dphi*_min vs deta (0 = before cut, 1 = after cut)
    TH2F* mControlConvResoncances = nullptr;  // control histograms for cuts on conversions and resonances
  } qa;

  // HistogramRegistry registry{"qa", true, {
  //   {"yields", "centrality vs pT vs eta",  {HistogramType::kTH3F, { {100, 0, 100, "centrality"}, {40, 0, 20, "p_{T}"}, {100, -2, 2, "#eta"} }}},
  //   {"etaphi", "centrality vs eta vs phi", {HistogramType::kTH3F, { {100, 0, 100, "centrality"}, {100, -2, 2, "#eta"}, {200, 0, 2 * M_PI, "#varphi"} }}}
  // }};

  OutputObj<TH3F> yields{TH3F("yields", "centrality vs pT vs eta", 100, 0, 100, 40, 0, 20, 100, -2, 2)};
  OutputObj<TH3F> etaphi{TH3F("etaphi", "centrality vs eta vs phi", 100, 0, 100, 100, -2, 2, 200, 0, 2 * M_PI)};

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
      cfg.mPairCuts = true;
    }

    // --- OBJECT INIT ---
    same.setObject(new CorrelationContainer("sameEvent", "sameEvent", "NumberDensityPhiCentralityVtx", binning));
    mixed.setObject(new CorrelationContainer("mixedEvent", "mixedEvent", "NumberDensityPhiCentralityVtx", binning));
    //qaOutput.setObject(new TDirectory("qa", "qa"));

    if (cfgTwoTrackCut > 0) {
      qa.mTwoTrackDistancePt[0] = new TH3F("TwoTrackDistancePt[0]", ";#Delta#eta;#Delta#varphi^{*}_{min};#Delta p_{T}", 100, -0.15, 0.15, 100, -0.05, 0.05, 20, 0, 10);
      qa.mTwoTrackDistancePt[1] = (TH3F*)qa.mTwoTrackDistancePt[0]->Clone("TwoTrackDistancePt[1]");
      //qaOutput->Add(qa.mTwoTrackDistancePt[0]);
      //qaOutput->Add(qa.mTwoTrackDistancePt[1]);
    }

    if (cfg.mPairCuts) {
      qa.mControlConvResoncances = new TH2F("ControlConvResoncances", ";id;delta mass", 6, -0.5, 5.5, 500, -0.5, 0.5);
      //qaOutput->Add(qa.mControlConvResoncances);
    }
  }

  // Version with explicit nested loop
  void process(soa::Filtered<aod::CFCollisions>::iterator const& collision, soa::Filtered<aod::CFTracks> const& tracks)
  {
    LOGF(info, "Tracks for collision: %d | Vertex: %.1f | V0M: %.1f", tracks.size(), collision.posZ(), collision.centV0M());

    const auto centrality = collision.centV0M();

    same->fillEvent(centrality, CorrelationContainer::kCFStepReconstructed);

    int bSign = 1; // TODO magnetic field from CCDB

    for (auto& track1 : tracks) {

      // LOGF(info, "Track %f | %f | %f  %d %d", track1.eta(), track1.phi(), track1.pt(), track1.isGlobalTrack(), track1.isGlobalTrackSDD());

      // control histograms
      // ((TH3*) (registry.get("yields").get()))->Fill(centrality, track1.pt(), track1.eta());
      // ((TH3*) (registry.get("etaphi").get()))->Fill(centrality, track1.eta(), track1.phi());
      yields->Fill(centrality, track1.pt(), track1.eta());
      etaphi->Fill(centrality, track1.eta(), track1.phi());

      if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.charge() < 0) {
        continue;
      }

      same->getTriggerHist()->Fill(CorrelationContainer::kCFStepReconstructed, track1.pt(), centrality, collision.posZ());
      //mixed->getTriggerHist()->Fill(eventValues, CorrelationContainer::kCFStepReconstructed);

      for (auto& track2 : tracks) {
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

        if (cfg.mPairCuts && conversionCuts(track1, track2)) {
          continue;
        }

        if (cfgTwoTrackCut > 0 && twoTrackCut(track1, track2, bSign)) {
          continue;
        }

        float deltaPhi = track1.phi() - track2.phi();
        if (deltaPhi > 1.5 * TMath::Pi())
          deltaPhi -= TMath::TwoPi();
        if (deltaPhi < -0.5 * TMath::Pi())
          deltaPhi += TMath::TwoPi();

        same->getPairHist()->Fill(CorrelationContainer::kCFStepReconstructed,
                                  track1.eta() - track2.eta(), track2.pt(), track1.pt(), centrality, deltaPhi, collision.posZ());
        //mixed->getPairHist()->Fill(values, CorrelationContainer::kCFStepReconstructed);
      }
    }
  }

  template <typename T>
  bool conversionCuts(T const& track1, T const& track2)
  {
    // skip if like sign
    if (track1.charge() * track2.charge() > 0) {
      return false;
    }

    bool decision = false;

    if (conversionCut(track1, track2, Photon, cfgPairCutPhoton)) {
      decision = true;
    }
    if (conversionCut(track1, track2, K0, cfgPairCutK0)) {
      decision = true;
    }
    if (conversionCut(track1, track2, Lambda, cfgPairCutLambda)) {
      decision = true;
    }
    if (conversionCut(track2, track1, Lambda, cfgPairCutLambda)) {
      decision = true;
    }
    if (conversionCut(track1, track2, Phi, cfgPairCutPhi)) {
      decision = true;
    }
    if (conversionCut(track1, track2, Rho, cfgPairCutRho)) {
      decision = true;
    }

    return decision;
  }

  template <typename T>
  bool conversionCut(T const& track1, T const& track2, PairCuts conv, double cut)
  {
    //LOGF(info, "pt is %f %f", track1.pt(), track2.pt());

    if (cut < 0) {
      return false;
    }

    double massD1, massD2, massM;

    switch (conv) {
      case Photon:
        massD1 = 0.51e-3;
        massD2 = 0.51e-3;
        massM = 0;
        break;
      case K0:
        massD1 = 0.1396;
        massD2 = 0.1396;
        massM = 0.4976;
        break;
      case Lambda:
        massD1 = 0.9383;
        massD2 = 0.1396;
        massM = 1.115;
        break;
      case Phi:
        massD1 = 0.4937;
        massD2 = 0.4937;
        massM = 1.019;
        break;
      case Rho:
        massD1 = 0.1396;
        massD2 = 0.1396;
        massM = 0.770;
        break;
    }

    auto massC = getInvMassSquaredFast(track1, massD1, track2, massD2);

    if (TMath::Abs(massC - massM * massM) > cut * 5) {
      return false;
    }

    massC = getInvMassSquared(track1, massD1, track2, massD2);
    qa.mControlConvResoncances->Fill(static_cast<int>(conv), massC - massM * massM);
    if (massC > (massM - cut) * (massM - cut) && massC < (massM + cut) * (massM + cut)) {
      return true;
    }

    return false;
  }

  template <typename T>
  double getInvMassSquared(T const& track1, double m0_1, T const& track2, double m0_2)
  {
    // calculate inv mass squared
    // same can be achieved, but with more computing time with
    /*TLorentzVector photon, p1, p2;
     p1.SetPtEtaPhiM(triggerParticle->Pt(), triggerEta, triggerParticle->Phi(), 0.510e-3);
     p2.SetPtEtaPhiM(particle->Pt(), eta[j], particle->Phi(), 0.510e-3);
     photon = p1+p2;
     photon.M()*/

    float tantheta1 = 1e10;

    if (track1.eta() < -1e-10 || track1.eta() > 1e-10) {
      float expTmp = TMath::Exp(-track1.eta());
      tantheta1 = 2.0 * expTmp / (1.0 - expTmp * expTmp);
    }

    float tantheta2 = 1e10;
    if (track2.eta() < -1e-10 || track2.eta() > 1e-10) {
      float expTmp = TMath::Exp(-track2.eta());
      tantheta2 = 2.0 * expTmp / (1.0 - expTmp * expTmp);
    }

    float e1squ = m0_1 * m0_1 + track1.pt() * track1.pt() * (1.0 + 1.0 / tantheta1 / tantheta1);
    float e2squ = m0_2 * m0_2 + track2.pt() * track2.pt() * (1.0 + 1.0 / tantheta2 / tantheta2);

    float mass2 = m0_1 * m0_1 + m0_2 * m0_2 + 2 * (TMath::Sqrt(e1squ * e2squ) - (track1.pt() * track2.pt() * (TMath::Cos(track1.phi() - track2.phi()) + 1.0 / tantheta1 / tantheta2)));

    // Printf(Form("%f %f %f %f %f %f %f %f %f", pt1, eta1, phi1, pt2, eta2, phi2, m0_1, m0_2, mass2));

    return mass2;
  }

  template <typename T>
  double getInvMassSquaredFast(T const& track1, double m0_1, T const& track2, double m0_2)
  {
    // calculate inv mass squared approximately

    const float eta1 = track1.eta();
    const float eta2 = track2.eta();
    const float phi1 = track1.phi();
    const float phi2 = track2.phi();
    const float pt1 = track1.pt();
    const float pt2 = track2.pt();

    float tantheta1 = 1e10;

    if (eta1 < -1e-10 || eta1 > 1e-10) {
      float expTmp = 1.0 - eta1 + eta1 * eta1 / 2 - eta1 * eta1 * eta1 / 6 + eta1 * eta1 * eta1 * eta1 / 24;
      tantheta1 = 2.0 * expTmp / (1.0 - expTmp * expTmp);
    }

    float tantheta2 = 1e10;
    if (eta2 < -1e-10 || eta2 > 1e-10) {
      float expTmp = 1.0 - eta2 + eta2 * eta2 / 2 - eta2 * eta2 * eta2 / 6 + eta2 * eta2 * eta2 * eta2 / 24;
      tantheta2 = 2.0 * expTmp / (1.0 - expTmp * expTmp);
    }

    float e1squ = m0_1 * m0_1 + pt1 * pt1 * (1.0 + 1.0 / tantheta1 / tantheta1);
    float e2squ = m0_2 * m0_2 + pt2 * pt2 * (1.0 + 1.0 / tantheta2 / tantheta2);

    // fold onto 0...pi
    float deltaPhi = TMath::Abs(phi1 - phi2);
    while (deltaPhi > TMath::TwoPi()) {
      deltaPhi -= TMath::TwoPi();
    }
    if (deltaPhi > TMath::Pi()) {
      deltaPhi = TMath::TwoPi() - deltaPhi;
    }

    float cosDeltaPhi = 0;
    if (deltaPhi < TMath::Pi() / 3) {
      cosDeltaPhi = 1.0 - deltaPhi * deltaPhi / 2 + deltaPhi * deltaPhi * deltaPhi * deltaPhi / 24;
    } else if (deltaPhi < 2 * TMath::Pi() / 3) {
      cosDeltaPhi = -(deltaPhi - TMath::Pi() / 2) + 1.0 / 6 * TMath::Power((deltaPhi - TMath::Pi() / 2), 3);
    } else {
      cosDeltaPhi = -1.0 + 1.0 / 2.0 * (deltaPhi - TMath::Pi()) * (deltaPhi - TMath::Pi()) - 1.0 / 24.0 * TMath::Power(deltaPhi - TMath::Pi(), 4);
    }

    double mass2 = m0_1 * m0_1 + m0_2 * m0_2 + 2 * (TMath::Sqrt(e1squ * e2squ) - (pt1 * pt2 * (cosDeltaPhi + 1.0 / tantheta1 / tantheta2)));

    //   Printf(Form("%f %f %f %f %f %f %f %f %f", pt1, eta1, phi1, pt2, eta2, phi2, m0_1, m0_2, mass2));

    return mass2;
  }

  template <typename T>
  bool twoTrackCut(T const& track1, T const& track2, int bSign)
  {
    // the variables & cuthave been developed by the HBT group
    // see e.g. https://indico.cern.ch/materialDisplay.py?contribId=36&sessionId=6&materialId=slides&confId=142700

    auto deta = track1.eta() - track2.eta();

    // optimization
    if (TMath::Abs(deta) < cfgTwoTrackCut * 2.5 * 3) {
      // check first boundaries to see if is worth to loop and find the minimum
      float dphistar1 = getDPhiStar(track1, track2, cfgTwoTrackCutMinRadius, bSign);
      float dphistar2 = getDPhiStar(track1, track2, 2.5, bSign);

      const float kLimit = cfgTwoTrackCut * 3;

      if (TMath::Abs(dphistar1) < kLimit || TMath::Abs(dphistar2) < kLimit || dphistar1 * dphistar2 < 0) {
        float dphistarminabs = 1e5;
        float dphistarmin = 1e5;
        for (Double_t rad = cfgTwoTrackCutMinRadius; rad < 2.51; rad += 0.01) {
          float dphistar = getDPhiStar(track1, track2, rad, bSign);

          float dphistarabs = TMath::Abs(dphistar);

          if (dphistarabs < dphistarminabs) {
            dphistarmin = dphistar;
            dphistarminabs = dphistarabs;
          }
        }

        qa.mTwoTrackDistancePt[0]->Fill(deta, dphistarmin, TMath::Abs(track1.pt() - track2.pt()));

        if (dphistarminabs < cfgTwoTrackCut && TMath::Abs(deta) < cfgTwoTrackCut) {
          //Printf("Removed track pair %ld %ld with %f %f %f %f %d %f %f %d %d", track1.index(), track2.index(), deta, dphistarminabs, track1.phi2(), track1.pt(), track1.charge(), track2.phi2(), track2.pt(), track2.charge(), bSign);
          return true;
        }

        qa.mTwoTrackDistancePt[1]->Fill(deta, dphistarmin, TMath::Abs(track1.pt() - track2.pt()));
      }
    }

    return false;
  }

  template <typename T>
  float getDPhiStar(T const& track1, T const& track2, float radius, float bSign)
  {
    //
    // calculates dphistar
    //

    auto phi1 = track1.phi();
    auto pt1 = track1.pt();
    auto charge1 = track1.charge();

    auto phi2 = track2.phi();
    auto pt2 = track2.pt();
    auto charge2 = track2.charge();

    float dphistar = phi1 - phi2 - charge1 * bSign * TMath::ASin(0.075 * radius / pt1) + charge2 * bSign * TMath::ASin(0.075 * radius / pt2);

    static const Double_t kPi = TMath::Pi();

    if (dphistar > kPi) {
      dphistar = kPi * 2 - dphistar;
    }
    if (dphistar < -kPi) {
      dphistar = -kPi * 2 - dphistar;
    }
    if (dphistar > kPi) { // might look funny but is needed
      dphistar = kPi * 2 - dphistar;
    }

    return dphistar;
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<CorrelationTask>("correlation-task")};
}
