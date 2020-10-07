// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ANALYSIS_PAIRCUTS_H
#define O2_ANALYSIS_PAIRCUTS_H

#include <cmath>
#include <TH2F.h>
#include <TH3F.h>

// Functions which cut on particle pairs (decays, conversions, two-track cuts)
//
// Author: Jan Fiete Grosse-Oetringhaus

class PairCuts
{
 public:
  enum Particle { Photon = 0,
                  K0,
                  Lambda,
                  Phi,
                  Rho,
                  ParticlesLastEntry };

  void SetPairCut(Particle particle, float cut)
  {
    mCuts[particle] = cut;
    if (mControlConvResoncances == nullptr) {
      mControlConvResoncances = new TH2F("ControlConvResoncances", ";id;delta mass", 6, -0.5, 5.5, 500, -0.5, 0.5);
    }
  }

  void SetTwoTrackCuts(float distance = 0.02f, float radius = 0.8f)
  {
    mTwoTrackDistance = distance;
    mTwoTrackRadius = radius;

    if (mTwoTrackDistancePt[0] == nullptr) {
      mTwoTrackDistancePt[0] = new TH3F("TwoTrackDistancePt[0]", ";#Delta#eta;#Delta#varphi^{*}_{min};#Delta p_{T}", 100, -0.15, 0.15, 100, -0.05, 0.05, 20, 0, 10);
      mTwoTrackDistancePt[1] = (TH3F*)mTwoTrackDistancePt[0]->Clone("TwoTrackDistancePt[1]");
    }
  }

  template <typename T>
  bool conversionCuts(T const& track1, T const& track2);

  template <typename T>
  bool twoTrackCut(T const& track1, T const& track2, int bSign);

 protected:
  float mCuts[ParticlesLastEntry] = {-1};
  float mTwoTrackDistance = -1; // distance below which the pair is flagged as to be removed
  float mTwoTrackRadius = 0.8f; // radius at which the two track cuts are applied

  TH2F* mControlConvResoncances = nullptr;  // control histograms for cuts on conversions and resonances
  TH3F* mTwoTrackDistancePt[2] = {nullptr}; // control histograms for two-track efficiency study: dphi*_min vs deta (0 = before cut, 1 = after cut)

  template <typename T>
  bool conversionCut(T const& track1, T const& track2, Particle conv, double cut);

  template <typename T>
  double getInvMassSquared(T const& track1, double m0_1, T const& track2, double m0_2);

  template <typename T>
  double getInvMassSquaredFast(T const& track1, double m0_1, T const& track2, double m0_2);

  template <typename T>
  float getDPhiStar(T const& track1, T const& track2, float radius, float bSign);
};

template <typename T>
bool PairCuts::conversionCuts(T const& track1, T const& track2)
{
  // skip if like sign
  if (track1.charge() * track2.charge() > 0) {
    return false;
  }

  for (int i = 0; i < static_cast<int>(ParticlesLastEntry); i++) {
    Particle particle = static_cast<Particle>(i);
    if (mCuts[i] > 0) {
      if (conversionCut(track1, track2, particle, mCuts[i])) {
        return true;
      }
      if (particle == Lambda) {
        if (conversionCut(track2, track1, particle, mCuts[i])) {
          return true;
        }
      }
    }
  }

  return false;
}

template <typename T>
bool PairCuts::twoTrackCut(T const& track1, T const& track2, int bSign)
{
  // the variables & cut have been developed in Run 1 by the CF - HBT group
  //
  // Parameters:
  //   bSign: sign of B field

  auto deta = track1.eta() - track2.eta();

  // optimization
  if (TMath::Abs(deta) < mTwoTrackDistance * 2.5 * 3) {
    // check first boundaries to see if is worth to loop and find the minimum
    float dphistar1 = getDPhiStar(track1, track2, mTwoTrackRadius, bSign);
    float dphistar2 = getDPhiStar(track1, track2, 2.5, bSign);

    const float kLimit = mTwoTrackDistance * 3;

    if (TMath::Abs(dphistar1) < kLimit || TMath::Abs(dphistar2) < kLimit || dphistar1 * dphistar2 < 0) {
      float dphistarminabs = 1e5;
      float dphistarmin = 1e5;
      for (Double_t rad = mTwoTrackRadius; rad < 2.51; rad += 0.01) {
        float dphistar = getDPhiStar(track1, track2, rad, bSign);

        float dphistarabs = TMath::Abs(dphistar);

        if (dphistarabs < dphistarminabs) {
          dphistarmin = dphistar;
          dphistarminabs = dphistarabs;
        }
      }

      mTwoTrackDistancePt[0]->Fill(deta, dphistarmin, TMath::Abs(track1.pt() - track2.pt()));

      if (dphistarminabs < mTwoTrackDistance && TMath::Abs(deta) < mTwoTrackDistance) {
        //LOGF(debug, "Removed track pair %ld %ld with %f %f %f %f %d %f %f %d %d", track1.index(), track2.index(), deta, dphistarminabs, track1.phi2(), track1.pt(), track1.charge(), track2.phi2(), track2.pt(), track2.charge(), bSign);
        return true;
      }

      mTwoTrackDistancePt[1]->Fill(deta, dphistarmin, TMath::Abs(track1.pt() - track2.pt()));
    }
  }

  return false;
}

template <typename T>
bool PairCuts::conversionCut(T const& track1, T const& track2, Particle conv, double cut)
{
  //LOGF(info, "pt is %f %f", track1.pt(), track2.pt());

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
    default:
      LOGF(fatal, "Particle now known");
      return false;
      break;
  }

  auto massC = getInvMassSquaredFast(track1, massD1, track2, massD2);

  if (TMath::Abs(massC - massM * massM) > cut * 5)
    return false;

  massC = getInvMassSquared(track1, massD1, track2, massD2);
  mControlConvResoncances->Fill(static_cast<int>(conv), massC - massM * massM);
  if (massC > (massM - cut) * (massM - cut) && massC < (massM + cut) * (massM + cut))
    return true;

  return false;
}

template <typename T>
double PairCuts::getInvMassSquared(T const& track1, double m0_1, T const& track2, double m0_2)
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

  // LOGF(debug, "%f %f %f %f %f %f %f %f %f", pt1, eta1, phi1, pt2, eta2, phi2, m0_1, m0_2, mass2);

  return mass2;
}

template <typename T>
double PairCuts::getInvMassSquaredFast(T const& track1, double m0_1, T const& track2, double m0_2)
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

  //LOGF(debug, "%f %f %f %f %f %f %f %f %f", pt1, eta1, phi1, pt2, eta2, phi2, m0_1, m0_2, mass2);

  return mass2;
}

template <typename T>
float PairCuts::getDPhiStar(T const& track1, T const& track2, float radius, float bSign)
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

  if (dphistar > M_PI)
    dphistar = M_PI * 2 - dphistar;
  if (dphistar < -M_PI)
    dphistar = -M_PI * 2 - dphistar;
  if (dphistar > M_PI) // might look funny but is needed
    dphistar = M_PI * 2 - dphistar;

  return dphistar;
}

#endif
