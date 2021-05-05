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

#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisDataModel/Centrality.h"
#include "AnalysisCore/CorrelationContainer.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

#include <TH1F.h>
#include <cmath>
#include <TDirectory.h>

namespace o2::aod
{
namespace hash
{
DECLARE_SOA_COLUMN(Bin, bin, int);
} // namespace hash
DECLARE_SOA_TABLE(Hashes, "AOD", "HASH", hash::Bin);

using Hash = Hashes::iterator;
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct HashTask {
  std::vector<float> vtxBins{-7.0f, -5.0f, -3.0f, -1.0f, 1.0f, 3.0f, 5.0f, 7.0f};
  std::vector<float> multBins{0.0f, 20.0f, 40.0f, 60.0f, 80.0f, 100.0f};
  Produces<aod::Hashes> hashes;

  // Calculate hash for an element based on 2 properties and their bins.
  int getHash(std::vector<float> const& vtxBins, std::vector<float> const& multBins, float vtx, float mult)
  {
    // underflow
    if (vtx < vtxBins[0]) {
      return -1;
    }
    if (mult < multBins[0]) {
      return -1;
    }

    for (int i = 1; i < vtxBins.size(); i++) {
      if (vtx < vtxBins[i]) {
        for (int j = 1; j < multBins.size(); j++) {
          if (mult < multBins[j]) {
            return i + j * (vtxBins.size() + 1);
          }
        }
      }
    }
    // overflow
    return -1;
  }

  void process(soa::Join<aod::Collisions, aod::Cents> const& collisions)
  {
    for (auto& collision : collisions) {
      int hash = getHash(vtxBins, multBins, collision.posZ(), collision.centV0M());
      LOGF(info, "Collision: %d (%f, %f) hash: %d", collision.index(), collision.posZ(), collision.centV0M(), hash);
      hashes(hash);
    }
  }
};

#define O2_DEFINE_CONFIGURABLE(NAME, TYPE, DEFAULT, HELP) Configurable<TYPE> NAME{#NAME, DEFAULT, HELP};

struct CorrelationTaskMixed {

  // Input definitions
  using myTracks = soa::Filtered<aod::Tracks>;

  // Filters
  Filter trackFilter = (aod::track::eta > -0.8f) && (aod::track::eta < 0.8f) && (aod::track::pt > 1.0f);

  // Output definitions
  OutputObj<CorrelationContainer> same{"sameEvent"};
  OutputObj<CorrelationContainer> mixed{"mixedEvent"};
  //OutputObj<TDirectory> qaOutput{"qa"};

  // Configuration
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

  ConfigurableAxis axisVertex{"axisVertex", {7, -7, 7}, "vertex axis for histograms"};
  ConfigurableAxis axisDeltaPhi{"axisDeltaPhi", {72, -M_PI / 2, M_PI / 2 * 3}, "delta phi axis for histograms"};
  ConfigurableAxis axisDeltaEta{"axisDeltaEta", {40, -2, 2}, "delta eta axis for histograms"};
  ConfigurableAxis axisPtTrigger{"axisPtTrigger", {VARIABLE_WIDTH, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0}, "pt trigger axis for histograms"};
  ConfigurableAxis axisPtAssoc{"axisPtAssoc", {VARIABLE_WIDTH, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}, "pt associated axis for histograms"};
  ConfigurableAxis axisMultiplicity{"axisMultiplicity", {VARIABLE_WIDTH, 0, 5, 10, 20, 30, 40, 50, 100.1}, "multiplicity / centrality axis for histograms"};

  ConfigurableAxis axisVertexEfficiency{"axisVertexEfficiency", {10, -10, 10}, "vertex axis for efficiency histograms"};
  ConfigurableAxis axisEtaEfficiency{"axisEtaEfficiency", {20, -1.0, 1.0}, "eta axis for efficiency histograms"};
  ConfigurableAxis axisPtEfficiency{"axisPtEfficiency", {VARIABLE_WIDTH, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0}, "pt axis for efficiency histograms"};

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

  void init(o2::framework::InitContext&)
  {
    // --- CONFIGURATION ---
    if (cfgPairCutPhoton > 0 || cfgPairCutK0 > 0 || cfgPairCutLambda > 0 || cfgPairCutPhi > 0 || cfgPairCutRho > 0) {
      cfg.mPairCuts = true;
    }

    // --- OBJECT INIT ---
    std::vector<AxisSpec> axisList = {{axisDeltaEta, "#Delta#eta"},
                                      {axisPtAssoc, "p_{T} (GeV/c)"},
                                      {axisPtTrigger, "p_{T} (GeV/c)"},
                                      {axisMultiplicity, "multiplicity / centrality"},
                                      {axisDeltaPhi, "#Delta#varphi (rad)"},
                                      {axisVertex, "z-vtx (cm)"},
                                      {axisEtaEfficiency, "#eta"},
                                      {axisPtEfficiency, "p_{T} (GeV/c)"},
                                      {axisVertexEfficiency, "z-vtx (cm)"}};
    same.setObject(new CorrelationContainer("sameEvent", "sameEvent", axisList));
    mixed.setObject(new CorrelationContainer("mixedEvent", "mixedEvent", axisList));

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

  void process(soa::Join<aod::Collisions, aod::Hashes, aod::EvSels, aod::Cents>& collisions, myTracks const& tracks)
  {
    int bSign = 1; // TODO magnetic field from CCDB
    const float pTCut = 2.0;

    collisions.bindExternalIndices(&tracks);
    auto tracksTuple = std::make_tuple(tracks);
    AnalysisDataProcessorBuilder::GroupSlicer slicer(collisions, tracksTuple);

    // Strictly upper categorised collisions, for 5 combinations per bin, skipping those in entry -1
    for (auto& [collision1, collision2] : selfCombinations("fBin", 5, -1, collisions, collisions)) {

      LOGF(info, "Collisions bin: %d pair: %d (%f), %d (%f)", collision1.bin(), collision1.index(), collision1.posZ(), collision2.index(), collision2.posZ());

      auto it1 = slicer.begin();
      auto it2 = slicer.begin();
      for (auto& slice : slicer) {
        if (slice.groupingElement().index() == collision1.index()) {
          it1 = slice;
          break;
        }
      }
      for (auto& slice : slicer) {
        if (slice.groupingElement().index() == collision2.index()) {
          it2 = slice;
          break;
        }
      }

      auto tracks1 = std::get<myTracks>(it1.associatedTables());
      tracks1.bindExternalIndices(&collisions);
      auto tracks2 = std::get<myTracks>(it2.associatedTables());
      tracks2.bindExternalIndices(&collisions);

      //       LOGF(info, "Tracks: %d and %d entries", tracks1.size(), tracks2.size());

      for (auto& track1 : tracks1) {

        if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.sign() < 0) {
          continue;
        }

        //LOGF(info, "TRACK %f %f | %f %f | %f %f", track1.eta(), track1.eta(), track1.phi(), track1.phi(), track1.pt(), track1.pt());

        mixed->getTriggerHist()->Fill(CorrelationContainer::kCFStepReconstructed, track1.pt(), collision1.centV0M(), collision1.posZ());

        for (auto& track2 : tracks2) {

          if (cfgAssociatedCharge != 0 && cfgAssociatedCharge * track2.sign() < 0) {
            continue;
          }
          if (cfgPairCharge != 0 && cfgPairCharge * track1.sign() * track2.sign() < 0) {
            continue;
          }

          if (cfg.mPairCuts && conversionCuts(track1, track2)) {
            continue;
          }

          if (cfgTwoTrackCut > 0 && twoTrackCut(track1, track2, bSign)) {
            continue;
          }

          float deltaPhi = track1.phi() - track2.phi();
          if (deltaPhi > 1.5 * M_PI) {
            deltaPhi -= M_PI * 2;
          }
          if (deltaPhi < -0.5 * M_PI) {
            deltaPhi += M_PI * 2;
          }

          mixed->getPairHist()->Fill(CorrelationContainer::kCFStepReconstructed,
                                     track1.eta() - track2.eta(), track2.pt(), track1.pt(), collision1.centV0M(), deltaPhi, collision1.posZ());
        }
      }
    }
  }

  template <typename T>
  bool conversionCuts(T const& track1, T const& track2)
  {
    // skip if like sign
    if (track1.sign() * track2.sign() > 0) {
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
    while (deltaPhi > M_PI * 2) {
      deltaPhi -= M_PI * 2;
    }
    if (deltaPhi > M_PI) {
      deltaPhi = M_PI * 2 - deltaPhi;
    }

    float cosDeltaPhi = 0;
    if (deltaPhi < M_PI / 3) {
      cosDeltaPhi = 1.0 - deltaPhi * deltaPhi / 2 + deltaPhi * deltaPhi * deltaPhi * deltaPhi / 24;
    } else if (deltaPhi < 2 * M_PI / 3) {
      cosDeltaPhi = -(deltaPhi - M_PI / 2) + 1.0 / 6 * TMath::Power((deltaPhi - M_PI / 2), 3);
    } else {
      cosDeltaPhi = -1.0 + 1.0 / 2.0 * (deltaPhi - M_PI) * (deltaPhi - M_PI) - 1.0 / 24.0 * TMath::Power(deltaPhi - M_PI, 4);
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
          //Printf("Removed track pair %ld %ld with %f %f %f %f %d %f %f %d %d", track1.index(), track2.index(), deta, dphistarminabs, track1.phi(), track1.pt(), track1.sign(), track2.phi(), track2.pt(), track2.sign(), bSign);
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
    auto charge1 = track1.sign();

    auto phi2 = track2.phi();
    auto pt2 = track2.pt();
    auto charge2 = track2.sign();

    float dphistar = phi1 - phi2 - charge1 * bSign * TMath::ASin(0.075 * radius / pt1) + charge2 * bSign * TMath::ASin(0.075 * radius / pt2);

    if (dphistar > M_PI) {
      dphistar = M_PI * 2 - dphistar;
    }
    if (dphistar < -M_PI) {
      dphistar = -M_PI * 2 - dphistar;
    }
    if (dphistar > M_PI) { // might look funny but is needed
      dphistar = M_PI * 2 - dphistar;
    }

    return dphistar;
  }

  //   template<typename... Ts>
  //   unsigned int getFilterBit(soa::Table<Ts...>::iterator const& track)
  //   {
  //     if constexpr(!has_type_v<aod::track::X, pack<Ts...>>)
  //       static_assert("Need to pass aod::track");
  //
  //
  // //     LOGF(info, "pt %f", track1.pt());
  //     return false;
  //   }

  //   float getInvMassSquared(float pt1, float eta1, float phi1, float pt2, float eta2, float phi2, float m0_1, float m0_2)
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<HashTask>(cfgc),
    adaptAnalysisTask<CorrelationTaskMixed>(cfgc)};
}
