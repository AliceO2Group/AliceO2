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

#include "Framework/Logger.h"
#include "Framework/HistogramRegistry.h"

// Functions which cut on particle pairs (decays, conversions, two-track cuts)
//
// Author: Jan Fiete Grosse-Oetringhaus

using namespace o2;
using namespace o2::framework;

class PairCuts
{
 public:
  enum Particle { Photon = 0,
                  K0,
                  Lambda,
                  Phi,
                  Rho,
                  ParticlesLastEntry };

  void SetHistogramRegistry(HistogramRegistry* registry) { histogramRegistry = registry; }

  void SetPairCut(Particle particle, float cut)
  {
    LOGF(info, "Enabled pair cut for %d with value %f", static_cast<int>(particle), cut);
    mCuts[particle] = cut;
    if (histogramRegistry != nullptr && histogramRegistry->contains(HIST("ControlConvResonances")) == false) {
      histogramRegistry->add("ControlConvResonances", "", {HistType::kTH2F, {{6, -0.5, 5.5, "id"}, {500, -0.5, 0.5, "delta mass"}}});
    }
  }

  void SetTwoTrackCuts(float distance = 0.02f, float radius = 0.8f)
  {
    LOGF(info, "Enabled two-track cut with distance %f and radius %f", distance, radius);
    mTwoTrackDistance = distance;
    mTwoTrackRadius = radius;

    if (histogramRegistry != nullptr && histogramRegistry->contains(HIST("TwoTrackDistancePt_0")) == false) {
      histogramRegistry->add("TwoTrackDistancePt_0", "", {HistType::kTH3F, {{100, -0.15, 0.15, "#Delta#eta"}, {100, -0.05, 0.05, "#Delta#varphi^{*}_{min}"}, {20, 0, 10, "#Delta p_{T}"}}});
      histogramRegistry->addClone("TwoTrackDistancePt_0", "TwoTrackDistancePt_1");
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

  HistogramRegistry* histogramRegistry = nullptr; // if set, control histograms are stored here

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
  if (track1.sign() * track2.sign() > 0) {
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
  if (std::fabs(deta) < mTwoTrackDistance * 2.5 * 3) {
    // check first boundaries to see if is worth to loop and find the minimum
    float dphistar1 = getDPhiStar(track1, track2, mTwoTrackRadius, bSign);
    float dphistar2 = getDPhiStar(track1, track2, 2.5, bSign);

    const float kLimit = mTwoTrackDistance * 3;

    if (std::fabs(dphistar1) < kLimit || std::fabs(dphistar2) < kLimit || dphistar1 * dphistar2 < 0) {
      float dphistarminabs = 1e5;
      float dphistarmin = 1e5;
      for (Double_t rad = mTwoTrackRadius; rad < 2.51; rad += 0.01) {
        float dphistar = getDPhiStar(track1, track2, rad, bSign);

        float dphistarabs = std::fabs(dphistar);

        if (dphistarabs < dphistarminabs) {
          dphistarmin = dphistar;
          dphistarminabs = dphistarabs;
        }
      }

      if (histogramRegistry != nullptr) {
        histogramRegistry->fill(HIST("TwoTrackDistancePt_0"), deta, dphistarmin, std::fabs(track1.pt() - track2.pt()));
      }

      if (dphistarminabs < mTwoTrackDistance && std::fabs(deta) < mTwoTrackDistance) {
        //LOGF(debug, "Removed track pair %ld %ld with %f %f %f %f %d %f %f %d %d", track1.index(), track2.index(), deta, dphistarminabs, track1.phi2(), track1.pt(), track1.sign(), track2.phi2(), track2.pt(), track2.sign(), bSign);
        return true;
      }

      if (histogramRegistry != nullptr) {
        histogramRegistry->fill(HIST("TwoTrackDistancePt_1"), deta, dphistarmin, std::fabs(track1.pt() - track2.pt()));
      }
    }
  }

  return false;
}

template <typename T>
bool PairCuts::conversionCut(T const& track1, T const& track2, Particle conv, double cut)
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
    default:
      LOGF(fatal, "Particle now known");
      return false;
      break;
  }

  auto massC = getInvMassSquaredFast(track1, massD1, track2, massD2);

  if (std::fabs(massC - massM * massM) > cut * 5) {
    return false;
  }

  massC = getInvMassSquared(track1, massD1, track2, massD2);

  if (histogramRegistry != nullptr) {
    histogramRegistry->fill(HIST("ControlConvResonances"), static_cast<int>(conv), massC - massM * massM);
  }

  if (massC > (massM - cut) * (massM - cut) && massC < (massM + cut) * (massM + cut)) {
    return true;
  }

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
    float expTmp = std::exp(-track1.eta());
    tantheta1 = 2.0 * expTmp / (1.0 - expTmp * expTmp);
  }

  float tantheta2 = 1e10;
  if (track2.eta() < -1e-10 || track2.eta() > 1e-10) {
    float expTmp = std::exp(-track2.eta());
    tantheta2 = 2.0 * expTmp / (1.0 - expTmp * expTmp);
  }

  float e1squ = m0_1 * m0_1 + track1.pt() * track1.pt() * (1.0 + 1.0 / tantheta1 / tantheta1);
  float e2squ = m0_2 * m0_2 + track2.pt() * track2.pt() * (1.0 + 1.0 / tantheta2 / tantheta2);

  float mass2 = m0_1 * m0_1 + m0_2 * m0_2 + 2 * (std::sqrt(e1squ * e2squ) - (track1.pt() * track2.pt() * (std::cos(track1.phi() - track2.phi()) + 1.0 / tantheta1 / tantheta2)));

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
  float deltaPhi = std::fabs(phi1 - phi2);
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
    cosDeltaPhi = -(deltaPhi - M_PI / 2) + 1.0 / 6 * std::pow((deltaPhi - M_PI / 2), 3);
  } else {
    cosDeltaPhi = -1.0 + 1.0 / 2.0 * (deltaPhi - M_PI) * (deltaPhi - M_PI) - 1.0 / 24.0 * std::pow(deltaPhi - M_PI, 4);
  }

  double mass2 = m0_1 * m0_1 + m0_2 * m0_2 + 2 * (std::sqrt(e1squ * e2squ) - (pt1 * pt2 * (cosDeltaPhi + 1.0 / tantheta1 / tantheta2)));

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
  auto charge1 = track1.sign();

  auto phi2 = track2.phi();
  auto pt2 = track2.pt();
  auto charge2 = track2.sign();

  float dphistar = phi1 - phi2 - charge1 * bSign * std::asin(0.075 * radius / pt1) + charge2 * bSign * std::asin(0.075 * radius / pt2);

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

#endif
