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
#include "Analysis/StepTHn.h"
#include "Analysis/CorrelationContainer.h"

#include <TH1F.h>
#include <cmath>
#include <TDirectory.h>

namespace o2::aod
{
namespace etaphi
{
DECLARE_SOA_COLUMN(Eta2, eta2, float);
DECLARE_SOA_COLUMN(Phi2, phi2, float);
DECLARE_SOA_COLUMN(Pt2, pt2, float);
} // namespace etaphi
DECLARE_SOA_TABLE(EtaPhi, "AOD", "ETAPHI",
                  etaphi::Eta2, etaphi::Phi2, etaphi::Pt2);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct ATask {
  Produces<aod::EtaPhi> etaphi;

  void process(aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      float eta = log(tan(0.25f * static_cast<float>(M_PI) - 0.5f * atan(track.tgl())));
      float phi = asin(track.snp()) + track.alpha() + static_cast<float>(M_PI);
      float pt = fabs(1.0f / track.signed1Pt());

      etaphi(eta, phi, pt);
    }
  }
};

#define O2_DEFINE_CONFIGURABLE(NAME, TYPE, DEFAULT, HELP) Configurable<TYPE> NAME{#NAME, DEFAULT, HELP};

struct CorrelationTask {

  // Input definitions
  using myTracks = soa::Join<aod::Tracks, aod::EtaPhi>;

  // Filters
#define MYFILTER
#ifdef MYFILTER
  Filter trackFilter = (aod::etaphi::eta2 > -0.8f) && (aod::etaphi::eta2 < 0.8f) && (aod::etaphi::pt2 > 1.0f);
#endif

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
    const char* binning =
      "vertex: -7, -5, -3, -1, 1, 3, 5, 7\n"
      "delta_phi: -1.570796, -1.483530, -1.396263, -1.308997, -1.221730, -1.134464, -1.047198, -0.959931, -0.872665, -0.785398, -0.698132, -0.610865, -0.523599, -0.436332, -0.349066, -0.261799, -0.174533, -0.087266, 0.0, 0.087266, 0.174533, 0.261799, 0.349066, 0.436332, 0.523599, 0.610865, 0.698132, 0.785398, 0.872665, 0.959931, 1.047198, 1.134464, 1.221730, 1.308997, 1.396263, 1.483530, 1.570796, 1.658063, 1.745329, 1.832596, 1.919862, 2.007129, 2.094395, 2.181662, 2.268928, 2.356194, 2.443461, 2.530727, 2.617994, 2.705260, 2.792527, 2.879793, 2.967060, 3.054326, 3.141593, 3.228859, 3.316126, 3.403392, 3.490659, 3.577925, 3.665191, 3.752458, 3.839724, 3.926991, 4.014257, 4.101524, 4.188790, 4.276057, 4.363323, 4.450590, 4.537856, 4.625123, 4.712389\n"
      "delta_eta: -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0\n"
      "p_t_assoc: 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\n"
      "p_t_trigger: 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0\n"
      "multiplicity: 0, 5, 10, 20, 30, 40, 50, 100.1\n"
      "eta: -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0\n"
      "p_t_leading: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5, 45.0, 45.5, 46.0, 46.5, 47.0, 47.5, 48.0, 48.5, 49.0, 49.5, 50.0\n"
      "p_t_leading_course: 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0\n"
      "p_t_eff: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0\n"
      "vertex_eff: -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10\n";

    if (cfgPairCutPhoton > 0 || cfgPairCutK0 > 0 || cfgPairCutLambda > 0 || cfgPairCutPhi > 0 || cfgPairCutRho > 0)
      cfg.mPairCuts = true;

    // --- OBJECT INIT ---
    same.setObject(new CorrelationContainer("sameEvent", "sameEvent", "NumberDensityPhiCentrality", binning));
    mixed.setObject(new CorrelationContainer("mixedEvent", "mixedEvent", "NumberDensityPhiCentrality", binning));
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
#ifdef MYFILTER
  void process(aod::Collision const& collision, aod::BCs const& bcs, aod::Run2V0s const& vzeros, soa::Filtered<soa::Join<aod::Tracks, aod::EtaPhi>> const& tracks)
#else
  void process(aod::Collision const& collision, aod::Run2V0s const& vzeros, soa::Join<aod::Tracks, aod::EtaPhi> const& tracks)
#endif
  {
    LOGF(info, "Tracks for collision: %d | Trigger mask: %lld", tracks.size(), collision.bc().triggerMask());
    //     for (auto& vzero : vzeros)
    //       if (vzero.bc() == collision.bc())
    //         LOGF(info, "V0: %f %f", vzero.adc()[0], vzero.adc()[1]);

    int bSign = 1; // TODO magnetic field from CCDB
    const float pTCut = 1.0;

    for (auto track1 = tracks.begin(); track1 != tracks.end(); ++track1) {

#ifdef MYFILTER
      if (track1.pt2() < pTCut)
        continue;
      if (track1.eta2() < -0.8 || track1.eta2() > 0.8)
        continue;
#endif

      if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.charge() < 0)
        continue;

      //LOGF(info, "TRACK %f %f | %f %f | %f %f", track1.eta(), track1.eta2(), track1.phi(), track1.phi2(), track1.pt(), track1.pt2());

      double eventValues[3];
      eventValues[0] = track1.pt2();
      eventValues[1] = 0; // collision.v0mult();
      eventValues[2] = collision.posZ();

      same->getEventHist()->Fill(eventValues, CorrelationContainer::kCFStepReconstructed);
      //mixed->getEventHist()->Fill(eventValues, CorrelationContainer::kCFStepReconstructed);

      for (auto track2 = track1 + 1; track2 != tracks.end(); ++track2) {
#ifdef MYFILTER
        if (track2.pt2() < pTCut)
          continue;
        if (track2.eta2() < -0.8 || track2.eta2() > 0.8)
          continue;
#endif

        if (cfgAssociatedCharge != 0 && cfgAssociatedCharge * track2.charge() < 0)
          continue;
        if (cfgPairCharge != 0 && cfgPairCharge * track1.charge() * track2.charge() < 0)
          continue;

        if (cfg.mPairCuts && conversionCuts(track1, track2))
          continue;

        if (cfgTwoTrackCut > 0 && twoTrackCut(track1, track2, bSign))
          continue;

        double values[6] = {0};

        values[0] = track1.eta2() - track2.eta2();
        values[1] = track1.pt2();
        values[2] = track2.pt2();
        values[3] = 0; // collision.v0mult();

        values[4] = track1.phi2() - track2.phi2();
        if (values[4] > 1.5 * TMath::Pi())
          values[4] -= TMath::TwoPi();
        if (values[4] < -0.5 * TMath::Pi())
          values[4] += TMath::TwoPi();

        values[5] = collision.posZ();

        same->getTrackHist()->Fill(values, CorrelationContainer::kCFStepReconstructed);
        //mixed->getTrackHist()->Fill(values, CorrelationContainer::kCFStepReconstructed);
      }
    }
  }

  // Version with combinations
  void process2(aod::Collision const& collision, soa::Filtered<soa::Join<aod::Tracks, aod::EtaPhi>> const& tracks)
  {
    LOGF(info, "Tracks for collision (Combination run): %d", tracks.size());

    int bSign = 1; // TODO magnetic field from CCDB

    for (auto track1 = tracks.begin(); track1 != tracks.end(); ++track1) {

      if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.charge() < 0)
        continue;

      //       LOGF(info, "TRACK %f %f | %f %f | %f %f", track1.eta(), track1.eta2(), track1.phi(), track1.phi2(), track1.pt(), track1.pt2());

      double eventValues[3];
      eventValues[0] = track1.pt2();
      eventValues[1] = 0; // collision.v0mult();
      eventValues[2] = collision.posZ();

      same->getEventHist()->Fill(eventValues, CorrelationContainer::kCFStepReconstructed);
      //mixed->getEventHist()->Fill(eventValues, CorrelationContainer::kCFStepReconstructed);
    }

    for (auto& [track1, track2] : combinations(tracks, tracks)) {
      //LOGF(info, "Combination %d %d", track1.index(), track2.index());

      if (cfgTriggerCharge != 0 && cfgTriggerCharge * track1.charge() < 0)
        continue;
      if (cfgAssociatedCharge != 0 && cfgAssociatedCharge * track2.charge() < 0)
        continue;
      if (cfgPairCharge != 0 && cfgPairCharge * track1.charge() * track2.charge() < 0)
        continue;

      if (cfg.mPairCuts && conversionCuts(track1, track2))
        continue;

      if (cfgTwoTrackCut > 0 && twoTrackCut(track1, track2, bSign))
        continue;

      double values[6] = {0};

      values[0] = track1.eta2() - track2.eta2();
      values[1] = track1.pt2();
      values[2] = track2.pt2();
      values[3] = 0; // collision.v0mult();

      values[4] = track1.phi2() - track2.phi2();
      if (values[4] > 1.5 * TMath::Pi())
        values[4] -= TMath::TwoPi();
      if (values[4] < -0.5 * TMath::Pi())
        values[4] += TMath::TwoPi();

      values[5] = collision.posZ();

      same->getTrackHist()->Fill(values, CorrelationContainer::kCFStepReconstructed);
      //mixed->getTrackHist()->Fill(values, CorrelationContainer::kCFStepReconstructed);
    }
  }

  template <typename T>
  bool conversionCuts(T const& track1, T const& track2)
  {
    // skip if like sign
    if (track1.charge() * track2.charge() > 0)
      return false;

    bool decision = false;

    if (conversionCut(track1, track2, Photon, cfgPairCutPhoton))
      decision = true;
    if (conversionCut(track1, track2, K0, cfgPairCutK0))
      decision = true;
    if (conversionCut(track1, track2, Lambda, cfgPairCutLambda))
      decision = true;
    if (conversionCut(track2, track1, Lambda, cfgPairCutLambda))
      decision = true;
    if (conversionCut(track1, track2, Phi, cfgPairCutPhi))
      decision = true;
    if (conversionCut(track1, track2, Rho, cfgPairCutRho))
      decision = true;

    return decision;
  }

  template <typename T>
  bool conversionCut(T const& track1, T const& track2, PairCuts conv, double cut)
  {
    //LOGF(info, "pt is %f %f", track1.pt2(), track2.pt2());

    if (cut < 0)
      return false;

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

    if (TMath::Abs(massC - massM * massM) > cut * 5)
      return false;

    massC = getInvMassSquared(track1, massD1, track2, massD2);
    qa.mControlConvResoncances->Fill(static_cast<int>(conv), massC - massM * massM);
    if (massC > (massM - cut) * (massM - cut) && massC < (massM + cut) * (massM + cut))
      return true;

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

    if (track1.eta2() < -1e-10 || track1.eta2() > 1e-10) {
      float expTmp = TMath::Exp(-track1.eta2());
      tantheta1 = 2.0 * expTmp / (1.0 - expTmp * expTmp);
    }

    float tantheta2 = 1e10;
    if (track2.eta2() < -1e-10 || track2.eta2() > 1e-10) {
      float expTmp = TMath::Exp(-track2.eta2());
      tantheta2 = 2.0 * expTmp / (1.0 - expTmp * expTmp);
    }

    float e1squ = m0_1 * m0_1 + track1.pt2() * track1.pt2() * (1.0 + 1.0 / tantheta1 / tantheta1);
    float e2squ = m0_2 * m0_2 + track2.pt2() * track2.pt2() * (1.0 + 1.0 / tantheta2 / tantheta2);

    float mass2 = m0_1 * m0_1 + m0_2 * m0_2 + 2 * (TMath::Sqrt(e1squ * e2squ) - (track1.pt2() * track2.pt2() * (TMath::Cos(track1.phi2() - track2.phi2()) + 1.0 / tantheta1 / tantheta2)));

    // Printf(Form("%f %f %f %f %f %f %f %f %f", pt1, eta1, phi1, pt2, eta2, phi2, m0_1, m0_2, mass2));

    return mass2;
  }

  template <typename T>
  double getInvMassSquaredFast(T const& track1, double m0_1, T const& track2, double m0_2)
  {
    // calculate inv mass squared approximately

    const float eta1 = track1.eta2();
    const float eta2 = track2.eta2();
    const float phi1 = track1.phi2();
    const float phi2 = track2.phi2();
    const float pt1 = track1.pt2();
    const float pt2 = track2.pt2();

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
    while (deltaPhi > TMath::TwoPi())
      deltaPhi -= TMath::TwoPi();
    if (deltaPhi > TMath::Pi())
      deltaPhi = TMath::TwoPi() - deltaPhi;

    float cosDeltaPhi = 0;
    if (deltaPhi < TMath::Pi() / 3)
      cosDeltaPhi = 1.0 - deltaPhi * deltaPhi / 2 + deltaPhi * deltaPhi * deltaPhi * deltaPhi / 24;
    else if (deltaPhi < 2 * TMath::Pi() / 3)
      cosDeltaPhi = -(deltaPhi - TMath::Pi() / 2) + 1.0 / 6 * TMath::Power((deltaPhi - TMath::Pi() / 2), 3);
    else
      cosDeltaPhi = -1.0 + 1.0 / 2.0 * (deltaPhi - TMath::Pi()) * (deltaPhi - TMath::Pi()) - 1.0 / 24.0 * TMath::Power(deltaPhi - TMath::Pi(), 4);

    double mass2 = m0_1 * m0_1 + m0_2 * m0_2 + 2 * (TMath::Sqrt(e1squ * e2squ) - (pt1 * pt2 * (cosDeltaPhi + 1.0 / tantheta1 / tantheta2)));

    //   Printf(Form("%f %f %f %f %f %f %f %f %f", pt1, eta1, phi1, pt2, eta2, phi2, m0_1, m0_2, mass2));

    return mass2;
  }

  template <typename T>
  bool twoTrackCut(T const& track1, T const& track2, int bSign)
  {
    // the variables & cuthave been developed by the HBT group
    // see e.g. https://indico.cern.ch/materialDisplay.py?contribId=36&sessionId=6&materialId=slides&confId=142700

    auto deta = track1.eta2() - track2.eta2();

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

        qa.mTwoTrackDistancePt[0]->Fill(deta, dphistarmin, TMath::Abs(track1.pt2() - track2.pt2()));

        if (dphistarminabs < cfgTwoTrackCut && TMath::Abs(deta) < cfgTwoTrackCut) {
          //Printf("Removed track pair %ld %ld with %f %f %f %f %d %f %f %d %d", track1.index(), track2.index(), deta, dphistarminabs, track1.phi2(), track1.pt2(), track1.charge(), track2.phi2(), track2.pt2(), track2.charge(), bSign);
          return true;
        }

        qa.mTwoTrackDistancePt[1]->Fill(deta, dphistarmin, TMath::Abs(track1.pt2() - track2.pt2()));
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

    auto phi1 = track1.phi2();
    auto pt1 = track1.pt2();
    auto charge1 = track1.charge();

    auto phi2 = track2.phi2();
    auto pt2 = track2.pt2();
    auto charge2 = track2.charge();

    float dphistar = phi1 - phi2 - charge1 * bSign * TMath::ASin(0.075 * radius / pt1) + charge2 * bSign * TMath::ASin(0.075 * radius / pt2);

    static const Double_t kPi = TMath::Pi();

    if (dphistar > kPi)
      dphistar = kPi * 2 - dphistar;
    if (dphistar < -kPi)
      dphistar = -kPi * 2 - dphistar;
    if (dphistar > kPi) // might look funny but is needed
      dphistar = kPi * 2 - dphistar;

    return dphistar;
  }

  //   template<typename... Ts>
  //   unsigned int getFilterBit(soa::Table<Ts...>::iterator const& track)
  //   {
  //     if constexpr(!has_type_v<aod::track::X, pack<Ts...>>)
  //       static_assert("Need to pass aod::track");
  //
  //
  // //     LOGF(info, "pt %f", track1.pt2());
  //     return false;
  //   }

  //   float getInvMassSquared(float pt1, float eta1, float phi1, float pt2, float eta2, float phi2, float m0_1, float m0_2)
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<ATask>("produce-etaphi"),
    adaptAnalysisTask<CorrelationTask>("correlation-task")};
}
