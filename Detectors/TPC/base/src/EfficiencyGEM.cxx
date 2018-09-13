// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file EfficiencyGEM.cxx
/// \brief Implementations for the model calculations + simulations of the GEM efficiencies
/// \author Viktor Ratza, University of Bonn, ratza@hiskp.uni-bonn.de

#include <cmath>

#include \
  "TPCBase/EfficiencyGEM.h"

using namespace o2::TPC;

EfficiencyGEM::EfficiencyGEM()
{
  // By default we use a standard pitch GEM
  setGeometry(1);
}

// Set GEM geometry (1 standard, 2 medium, 3 large)
void EfficiencyGEM::setGeometry(int geom)
{
  // Define GEM parameters and electric field configuration as it was used
  // in the simulations and for the fits of the calculations afterwards.
  // Additionally define the obtained tuning parameters from the fits.
  // All GEM geometry parameters are given in micrometers, all potentials in Volts
  // and all electric fields in Volts/cm.

  switch (geom) {
    case 1:
      mGeometryPitch = 140.0;
      mGeometryTuneEta1 = 9.7;
      mGeometryTuneEta2 = 55.3;
      mGeometryTuneDiffusion = 8.0;
      break;
    case 2:
      mGeometryPitch = 200.0;
      mGeometryTuneEta1 = 16.3;
      mGeometryTuneEta2 = 25.8;
      mGeometryTuneDiffusion = 3.8;
      break;
    case 3:
      mGeometryPitch = 280.0;
      mGeometryTuneEta1 = 25.2;
      mGeometryTuneEta2 = 25.5;
      mGeometryTuneDiffusion = 3.5;
      break;
  }

  mGeometryWidth = 2.0 * mGeometryPitch - mGeometryHoleDiameter;

  // Some parameters are constant for a fixed GEM pitch, so we evaluate them once after the pitch has been set or
  // altered
  mParamC1 = getParameterC1();
  mParamC2 = getParameterC2();
  mParamC3 = getParameterC3();
  mParamC4 = getParameterC4();
  mParamC5 = getParameterC5();
  mParamC6 = getParameterC6();
}

float EfficiencyGEM::getCollectionEfficiency(float ElecFieldRatioAbove)
{
  // Combined Fit 5.3
  float eta1 = mGeometryTuneEta1 * ElecFieldRatioAbove;
  float eta2 = sKappa * mElecFieldBelow / mElecFieldGEM;

  float paramC7Bar = getParameterC7Bar(eta1, eta2);
  float paramC8Bar = getParameterC8Bar(eta1, eta2);
  float paramC9Bar = getParameterC9Bar(eta1, eta2);

  return 2.0 * sPi * (paramC7Bar + paramC9Bar * eta1 + paramC8Bar * eta2) /
         (mParamC1 + mParamC3 * eta1 + mParamC2 * eta2);
}

float EfficiencyGEM::getExtractionEfficiency(float ElecFieldRatioBelow)
{
  // Combined Fit 5.3
  float eta1 = sKappa * mElecFieldAbove / mElecFieldGEM;
  float eta2 = mGeometryTuneEta2 * ElecFieldRatioBelow;

  float paramC7 = getParameterC7(eta1, eta2);
  float paramC8 = getParameterC8(eta1, eta2);
  float paramC9 = getParameterC9(eta1, eta2);

  return 2.0 * sPi * (paramC7 + paramC8 * eta1 + paramC9 * eta2) /
         (mParamC4 * mGeometryTuneDiffusion + mParamC5 * eta1 + mParamC6 * eta2);
}

const float EfficiencyGEM::getGeometryThickness()
{
  return mGeometryThickness;
}

float EfficiencyGEM::getParameterC1()
{
  return -getLambdaCathode();
}

float EfficiencyGEM::getParameterC2()
{
  return -getMu2Cathode();
}

float EfficiencyGEM::getParameterC3()
{
  return -getMu1Cathode();
}

float EfficiencyGEM::getParameterC4()
{
  return (getMu2Top(-mGeometryHoleDiameter / 2.0, mGeometryHoleDiameter / 2.0) + sPi * mGeometryHoleDiameter);
}

float EfficiencyGEM::getParameterC5()
{
  return -getMu2Top(-mGeometryHoleDiameter / 2.0, mGeometryHoleDiameter / 2.0);
}

float EfficiencyGEM::getParameterC6()
{
  return sPi * mGeometryHoleDiameter;
}

float EfficiencyGEM::getParameterC7(float eta1, float eta2)
{
  float IntXStart = -(mGeometryWidth + mGeometryHoleDiameter) / 4.0;
  float IntXEnd = getIntXEndBot(eta1, eta2);

  // Flip g1 <-> g2 for calculation since c7bar(g1->g2) = c7
  flipDistanceNextPrevStage();
  float result = getParameterC7BarFromX(IntXStart, IntXEnd);
  // Flip back g1<->g2 to initial condition
  flipDistanceNextPrevStage();

  return result;
}

float EfficiencyGEM::getParameterC8(float eta1, float eta2)
{
  float IntXStart = -(mGeometryWidth + mGeometryHoleDiameter) / 4.0;
  float IntXEnd = getIntXEndBot(eta1, eta2);

  flipDistanceNextPrevStage();
  float result = getParameterC8BarFromX(IntXStart, IntXEnd);
  flipDistanceNextPrevStage();

  return result;
}

float EfficiencyGEM::getParameterC9(float eta1, float eta2)
{
  float IntXStart = -(mGeometryWidth + mGeometryHoleDiameter) / 4.0;
  float IntXEnd = getIntXEndBot(eta1, eta2);

  flipDistanceNextPrevStage();
  float result = getParameterC9BarFromX(IntXStart, IntXEnd);
  flipDistanceNextPrevStage();

  return result;
}

float EfficiencyGEM::getParameterC7Bar(float eta1, float eta2)
{
  float IntXStart = -(mGeometryWidth + mGeometryHoleDiameter) / 4.0;
  float IntXEnd = getIntXEndTop(eta1, eta2);

  return getParameterC7BarFromX(IntXStart, IntXEnd);
}

float EfficiencyGEM::getParameterC8Bar(float eta1, float eta2)
{
  float IntXStart = -(mGeometryWidth + mGeometryHoleDiameter) / 4.0;
  float IntXEnd = getIntXEndTop(eta1, eta2);

  return getParameterC8BarFromX(IntXStart, IntXEnd);
}

float EfficiencyGEM::getParameterC9Bar(float eta1, float eta2)
{
  float IntXStart = -(mGeometryWidth + mGeometryHoleDiameter) / 4.0;
  float IntXEnd = getIntXEndTop(eta1, eta2);

  return getParameterC9BarFromX(IntXStart, IntXEnd);
}

float EfficiencyGEM::getParameterC7BarFromX(float IntXStart, float IntXEnd)
{
  return (-1.0 / (2.0 * sPi)) * (getLambdaCathode() + 2.0 * getMu2Top(IntXStart, IntXEnd));
}

float EfficiencyGEM::getParameterC8BarFromX(float IntXStart, float IntXEnd)
{
  return (-1.0 / (2.0 * sPi)) * (getMu2Cathode() - 2.0 * getMu2Top(IntXStart, IntXEnd));
}

float EfficiencyGEM::getParameterC9BarFromX(float IntXStart, float IntXEnd)
{
  return (-1.0 / (2.0 * sPi)) * (getMu1Cathode() + 4.0 * sPi * (IntXEnd - IntXStart));
}

void EfficiencyGEM::flipDistanceNextPrevStage()
{
  float mGeometryDistancePrevStageOld = mGeometryDistancePrevStage;
  mGeometryDistancePrevStage = mGeometryDistanceNextStage;
  mGeometryDistanceNextStage = mGeometryDistancePrevStageOld;
}

float EfficiencyGEM::getIntXEndBot(float eta1, float eta2)
{
  float result;

  if (eta2 <= getEta2Kink1(eta1)) {
    result = -(mGeometryHoleDiameter + mGeometryWidth) / 4.0;
  } else if (getEta2Kink1(eta1) < eta2 && eta2 < getEta2Kink2(eta1)) {
    result = -mGeometryPitch / 2.0 + sqrt(getHtop2() * (eta1 - 1.0) * (2.0 * sPi * eta2 + getHtop0() * (1.0 - eta1))) /
                                       (getHtop2() * (eta1 - 1.0));
  } else {
    result = -mGeometryHoleDiameter / 2.0;
  }

  return result;
}

float EfficiencyGEM::getIntXEndTop(float eta1, float eta2)
{
  float result;

  if (eta1 <= getEta1Kink1(eta2)) {
    result = -(mGeometryHoleDiameter + mGeometryWidth) / 4.0;
  } else if (getEta1Kink1(eta2) < eta1 && eta1 < getEta1Kink2(eta2)) {
    result = -mGeometryPitch / 2.0 + sqrt(getHtop2() * (eta2 - 1.0) * (2.0 * sPi * eta1 + getHtop0() * (1.0 - eta2))) /
                                       (getHtop2() * (eta2 - 1.0));
  } else {
    result = -mGeometryHoleDiameter / 2.0;
  }

  return result;
}

float EfficiencyGEM::getEta1Kink2(float eta2)
{
  return 1.0 / (2.0 * sPi) *
         ((mGeometryHoleDiameter - mGeometryPitch) * (mGeometryHoleDiameter - mGeometryPitch) / 4.0 * getHtop2() +
          getHtop0()) *
         (eta2 - 1.0);
}

float EfficiencyGEM::getEta1Kink1(float eta2)
{
  return -1.0 / (2.0 * sPi) * getHtop0() * (1.0 - eta2);
}

float EfficiencyGEM::getEta2Kink2(float eta1)
{
  return 1.0 / (2.0 * sPi) *
         ((mGeometryHoleDiameter - mGeometryPitch) * (mGeometryHoleDiameter - mGeometryPitch) / 4.0 * getHtop2() +
          getHtop0()) *
         (eta1 - 1.0);
}

float EfficiencyGEM::getEta2Kink1(float eta1)
{
  return -1.0 / (2.0 * sPi) * getHtop0() * (1.0 - eta1);
}

float EfficiencyGEM::getHtop0()
{
  float result = 0.0;

  for (int n = 2; n <= mNumberHoles; ++n) {
    result += getMu2TopFTaylorTerm0(n);
  }

  result += getMu2TopfTaylorTerm0();

  return result;
}

float EfficiencyGEM::getHtop2()
{
  float result = 0.0;

  for (int n = 2; n <= mNumberHoles; ++n) {
    result += getMu2TopFTaylorTerm2(n);
  }

  result += getMu2TopfTaylorTerm2();

  return result;
}

float EfficiencyGEM::getMu2TopfTaylorTerm2()
{
  float d = mGeometryThickness;
  float L = mGeometryHoleDiameter;
  float w = mGeometryWidth;

  return (12 * L + 4 * w) / d / ((9 * std::pow(L, 2)) + (6 * L * w) + 0.16e2 * std::pow(d, 0.2e1) + std::pow(w, 2)) /
           (std::pow(0.3e1 / 0.2e1 * L + w / 0.2e1, 0.2e1) * std::pow(d, -0.2e1) / 0.4e1 + 0.1e1) +
         (4 * L - 4 * w) / d / (std::pow(L, 2) - (2 * L * w) + 0.16e2 * std::pow(d, 0.2e1) + std::pow(w, 2)) /
           (std::pow(L / 0.2e1 - w / 0.2e1, 0.2e1) * std::pow(d, -0.2e1) / 0.4e1 + 0.1e1);
}

float EfficiencyGEM::getMu2TopFTaylorTerm2(int n)
{
  float d = mGeometryThickness;
  float L = mGeometryHoleDiameter;
  float w = mGeometryWidth;

  return -(8 * (n - 2) * L + 8 * (n - 1) * w - 4 * w - 4 * L) / d /
           ((4 * std::pow(L, 2) * std::pow(n, 2)) + (8 * L * std::pow(n, 2) * w) +
            (4 * std::pow(n, 2) * std::pow(w, 2)) - (20 * std::pow(L, 2) * n) - (32 * n * L * w) -
            (12 * n * std::pow(w, 2)) + (25 * std::pow(L, 2)) + (30 * L * w) + 0.16e2 * std::pow(d, 0.2e1) +
            (9 * std::pow(w, 2))) /
           (std::pow(((n - 2) * L) + ((n - 1) * w) - w / 0.2e1 - L / 0.2e1, 0.2e1) * std::pow(d, -0.2e1) / 0.4e1 +
            0.1e1) -
         (8 * (n - 2) * L + 8 * (n - 1) * w + 4 * w + 4 * L) / d /
           ((4 * std::pow(L, 2) * std::pow(n, 2)) + (8 * L * std::pow(n, 2) * w) +
            (4 * std::pow(n, 2) * std::pow(w, 2)) - (12 * std::pow(L, 2) * n) - (16 * n * L * w) -
            (4 * n * std::pow(w, 2)) + (9 * std::pow(L, 2)) + (6 * L * w) + 0.16e2 * std::pow(d, 0.2e1) +
            std::pow(w, 2)) /
           (std::pow(((n - 2) * L) + ((n - 1) * w) + w / 0.2e1 + L / 0.2e1, 0.2e1) * std::pow(d, -0.2e1) / 0.4e1 +
            0.1e1) +
         (8 * n * L + 8 * (n - 1) * w - 4 * w - 4 * L) / d /
           ((4 * std::pow(L, 2) * std::pow(n, 2)) + (8 * L * std::pow(n, 2) * w) +
            (4 * std::pow(n, 2) * std::pow(w, 2)) - (4 * std::pow(L, 2) * n) - (16 * n * L * w) -
            (12 * n * std::pow(w, 2)) + std::pow(L, 2) + (6 * L * w) + 0.16e2 * std::pow(d, 0.2e1) +
            (9 * std::pow(w, 2))) /
           (std::pow((n * L) + ((n - 1) * w) - w / 0.2e1 - L / 0.2e1, 0.2e1) * std::pow(d, -0.2e1) / 0.4e1 + 0.1e1) +
         (8 * n * L + 8 * (n - 1) * w + 4 * w + 4 * L) / d /
           ((4 * std::pow(L, 2) * std::pow(n, 2)) + (8 * L * std::pow(n, 2) * w) +
            (4 * std::pow(n, 2) * std::pow(w, 2)) + (4 * std::pow(L, 2) * n) - (4 * n * std::pow(w, 2)) +
            std::pow(L, 2) - (2 * L * w) + 0.16e2 * std::pow(d, 0.2e1) + std::pow(w, 2)) /
           (std::pow((n * L) + ((n - 1) * w) + w / 0.2e1 + L / 0.2e1, 0.2e1) * std::pow(d, -0.2e1) / 0.4e1 + 0.1e1);
}

float EfficiencyGEM::getMu2TopFTaylorTerm0(int n)
{
  float d = mGeometryThickness;
  float L = mGeometryHoleDiameter;
  float w = mGeometryWidth;

  return std::atan((((n - 2) * L) + ((n - 1) * w) - w / 0.2e1 - L / 0.2e1) / d / 0.2e1) +
         std::atan((((n - 2) * L) + ((n - 1) * w) + w / 0.2e1 + L / 0.2e1) / d / 0.2e1) -
         std::atan(((n * L) + ((n - 1) * w) - w / 0.2e1 - L / 0.2e1) / d / 0.2e1) -
         std::atan(((n * L) + ((n - 1) * w) + w / 0.2e1 + L / 0.2e1) / d / 0.2e1);
}

float EfficiencyGEM::getMu2TopfTaylorTerm0()
{
  float d = mGeometryThickness;
  float L = mGeometryHoleDiameter;
  float w = mGeometryWidth;

  return -std::atan((0.3e1 / 0.2e1 * L + w / 0.2e1) / d / 0.2e1) - std::atan((L / 0.2e1 - w / 0.2e1) / d / 0.2e1);
}

float EfficiencyGEM::getMu2Top(float IntXStart, float IntXEnd)
{
  float result = 0.0;

  for (int n = 2; n <= mNumberHoles; ++n) {
    result += getMu2TopF2(n, IntXStart, IntXEnd);
  }

  result += getMu2Topf2(IntXStart, IntXEnd);

  return result;
}

float EfficiencyGEM::getMu2Topf2(float IntXStart, float IntXEnd)
{
  float d = mGeometryThickness;
  float L = mGeometryHoleDiameter;

  return std::atan(((L - 2 * IntXStart) / d) / 0.2e1) * IntXStart -
         std::atan(((L - 2 * IntXStart) / d) / 0.2e1) * L / 0.2e1 +
         d * std::log((std::pow(L, 2) - 4 * L * IntXStart + 4 * std::pow(d, 2) + 4 * IntXStart * IntXStart)) / 0.2e1 +
         std::atan(((L + 2 * IntXStart) / d) / 0.2e1) * IntXStart +
         std::atan(((L + 2 * IntXStart) / d) / 0.2e1) * L / 0.2e1 -
         d * std::log((std::pow(L, 2) + 4 * L * IntXStart + 4 * std::pow(d, 2) + 4 * IntXStart * IntXStart)) / 0.2e1 -
         std::atan(((-2 * IntXEnd + L) / d) / 0.2e1) * IntXEnd +
         std::atan(((-2 * IntXEnd + L) / d) / 0.2e1) * L / 0.2e1 -
         d * std::log((std::pow(L, 2) - 4 * L * IntXEnd + 4 * std::pow(d, 2) + 4 * IntXEnd * IntXEnd)) / 0.2e1 -
         std::atan(((2 * IntXEnd + L) / d) / 0.2e1) * IntXEnd - std::atan(((2 * IntXEnd + L) / d) / 0.2e1) * L / 0.2e1 +
         d * std::log((std::pow(L, 2) + 4 * L * IntXEnd + 4 * std::pow(d, 2) + 4 * IntXEnd * IntXEnd)) / 0.2e1;
}

float EfficiencyGEM::getMu2TopF2(int n, float IntXStart, float IntXEnd)
{
  float d = mGeometryThickness;
  float L = mGeometryHoleDiameter;
  float w = mGeometryWidth;

  return std::atan(((n * L + n * w - w + 2 * IntXStart) / d) / 0.2e1) * IntXStart -
         std::atan(((n * L + n * w - w + 2 * IntXStart) / d) / 0.2e1) * w / 0.2e1 +
         std::atan(((n * L + n * w - w - 2 * IntXStart) / d) / 0.2e1) * IntXStart +
         std::atan(((n * L + n * w - w - 2 * IntXStart) / d) / 0.2e1) * w / 0.2e1 +
         std::atan(((n * L + n * w - 2 * L - w + 2 * IntXEnd) / d) / 0.2e1) * IntXEnd -
         std::atan(((n * L + n * w - 2 * L - w + 2 * IntXEnd) / d) / 0.2e1) * L -
         std::atan(((n * L + n * w - 2 * L - w + 2 * IntXEnd) / d) / 0.2e1) * w / 0.2e1 +
         std::atan(((n * L + n * w - 2 * L - w - 2 * IntXEnd) / d) / 0.2e1) * IntXEnd +
         std::atan(((n * L + n * w - 2 * L - w - 2 * IntXEnd) / d) / 0.2e1) * L +
         std::atan(((n * L + n * w - 2 * L - w - 2 * IntXEnd) / d) / 0.2e1) * w / 0.2e1 -
         std::atan(((n * L + n * w - w + 2 * IntXEnd) / d) / 0.2e1) * IntXEnd +
         std::atan(((n * L + n * w - w + 2 * IntXEnd) / d) / 0.2e1) * w / 0.2e1 -
         std::atan(((n * L + n * w - w - 2 * IntXEnd) / d) / 0.2e1) * IntXEnd -
         std::atan(((n * L + n * w - w - 2 * IntXEnd) / d) / 0.2e1) * w / 0.2e1 -
         std::atan(((n * L + n * w - 2 * L - w + 2 * IntXStart) / d) / 0.2e1) * IntXStart +
         std::atan(((n * L + n * w - 2 * L - w + 2 * IntXStart) / d) / 0.2e1) * L +
         std::atan(((n * L + n * w - 2 * L - w + 2 * IntXStart) / d) / 0.2e1) * w / 0.2e1 -
         std::atan(((n * L + n * w - 2 * L - w - 2 * IntXStart) / d) / 0.2e1) * IntXStart -
         std::atan(((n * L + n * w - 2 * L - w - 2 * IntXStart) / d) / 0.2e1) * L -
         std::atan(((n * L + n * w - 2 * L - w - 2 * IntXStart) / d) / 0.2e1) * w / 0.2e1 +
         std::atan(((n * L + n * w - 2 * L - w - 2 * IntXStart) / d) / 0.2e1) * n * L / 0.2e1 +
         std::atan(((n * L + n * w - 2 * L - w - 2 * IntXStart) / d) / 0.2e1) * n * w / 0.2e1 +
         std::atan(((n * L + n * w - w + 2 * IntXStart) / d) / 0.2e1) * n * w / 0.2e1 +
         std::atan(((n * L + n * w - w + 2 * IntXStart) / d) / 0.2e1) * n * L / 0.2e1 -
         std::atan(((n * L + n * w - w - 2 * IntXStart) / d) / 0.2e1) * n * L / 0.2e1 -
         std::atan(((n * L + n * w - w - 2 * IntXStart) / d) / 0.2e1) * n * w / 0.2e1 +
         std::atan(((n * L + n * w - 2 * L - w + 2 * IntXEnd) / d) / 0.2e1) * n * L / 0.2e1 +
         std::atan(((n * L + n * w - 2 * L - w + 2 * IntXEnd) / d) / 0.2e1) * n * w / 0.2e1 -
         std::atan(((n * L + n * w - 2 * L - w - 2 * IntXEnd) / d) / 0.2e1) * n * L / 0.2e1 -
         std::atan(((n * L + n * w - 2 * L - w - 2 * IntXEnd) / d) / 0.2e1) * n * w / 0.2e1 -
         std::atan(((n * L + n * w - w + 2 * IntXEnd) / d) / 0.2e1) * n * L / 0.2e1 -
         std::atan(((n * L + n * w - w + 2 * IntXEnd) / d) / 0.2e1) * n * w / 0.2e1 +
         std::atan(((n * L + n * w - w - 2 * IntXEnd) / d) / 0.2e1) * n * L / 0.2e1 +
         std::atan(((n * L + n * w - w - 2 * IntXEnd) / d) / 0.2e1) * n * w / 0.2e1 -
         std::atan(((n * L + n * w - 2 * L - w + 2 * IntXStart) / d) / 0.2e1) * n * L / 0.2e1 -
         std::atan(((n * L + n * w - 2 * L - w + 2 * IntXStart) / d) / 0.2e1) * n * w / 0.2e1 +
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) -
                       4 * std::pow(L, 2) * n - 6 * n * L * w - 4 * n * L * IntXEnd - 2 * n * std::pow(w, 2) -
                       4 * IntXEnd * w * n + 4 * std::pow(L, 2) + 4 * L * w + 8 * L * IntXEnd + 4 * std::pow(d, 2) +
                       std::pow(w, 2) + 4 * IntXEnd * w + 4 * IntXEnd * IntXEnd)) /
           0.2e1 -
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) -
                       2 * n * L * w + 4 * n * L * IntXStart - 2 * n * std::pow(w, 2) + 4 * w * IntXStart * n +
                       4 * std::pow(d, 2) + std::pow(w, 2) - 4 * w * IntXStart + 4 * IntXStart * IntXStart)) /
           0.2e1 -
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) -
                       2 * n * L * w - 4 * n * L * IntXEnd - 2 * n * std::pow(w, 2) - 4 * IntXEnd * w * n +
                       4 * std::pow(d, 2) + std::pow(w, 2) + 4 * IntXEnd * w + 4 * IntXEnd * IntXEnd)) /
           0.2e1 -
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) -
                       4 * std::pow(L, 2) * n - 6 * n * L * w - 4 * n * L * IntXStart - 2 * n * std::pow(w, 2) -
                       4 * w * IntXStart * n + 4 * std::pow(L, 2) + 4 * L * w + 8 * L * IntXStart + 4 * std::pow(d, 2) +
                       std::pow(w, 2) + 4 * w * IntXStart + 4 * IntXStart * IntXStart)) /
           0.2e1 +
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) -
                       4 * std::pow(L, 2) * n - 6 * n * L * w + 4 * n * L * IntXStart - 2 * n * std::pow(w, 2) +
                       4 * w * IntXStart * n + 4 * std::pow(L, 2) + 4 * L * w - 8 * L * IntXStart + 4 * std::pow(d, 2) +
                       std::pow(w, 2) - 4 * w * IntXStart + 4 * IntXStart * IntXStart)) /
           0.2e1 -
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) -
                       4 * std::pow(L, 2) * n - 6 * n * L * w + 4 * n * L * IntXEnd - 2 * n * std::pow(w, 2) +
                       4 * IntXEnd * w * n + 4 * std::pow(L, 2) + 4 * L * w - 8 * L * IntXEnd + 4 * std::pow(d, 2) +
                       std::pow(w, 2) - 4 * IntXEnd * w + 4 * IntXEnd * IntXEnd)) /
           0.2e1 +
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) -
                       2 * n * L * w - 4 * n * L * IntXStart - 2 * n * std::pow(w, 2) - 4 * w * IntXStart * n +
                       4 * std::pow(d, 2) + std::pow(w, 2) + 4 * w * IntXStart + 4 * IntXStart * IntXStart)) /
           0.2e1 +
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) -
                       2 * n * L * w + 4 * n * L * IntXEnd - 2 * n * std::pow(w, 2) + 4 * IntXEnd * w * n +
                       4 * std::pow(d, 2) + std::pow(w, 2) - 4 * IntXEnd * w + 4 * IntXEnd * IntXEnd)) /
           0.2e1;
}

float EfficiencyGEM::getMu1Cathode()
{
  float result = 0.0;

  for (int n = 2; n <= mNumberHoles; ++n) {
    result += getMu1CathodeF2(n);
  }

  result += getMu1Cathodef2();

  return result;
}

float EfficiencyGEM::getMu1Cathodef2()
{
  float d = mGeometryThickness;
  float L = mGeometryHoleDiameter;
  float w = mGeometryWidth;
  float g1 = mGeometryDistancePrevStage;

  return -0.3e1 / 0.2e1 * L * std::atan(((3 * L + w) / (-g1 + d)) / 0.2e1) +
         L * std::atan(((-w + L) / (-g1 + d)) / 0.2e1) / 0.2e1 - L * 0.3141592654e1 -
         std::atan(((3 * L + w) / (-g1 + d)) / 0.2e1) * w / 0.2e1 +
         d * std::log((9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) +
                       std::pow(w, 2))) /
           0.2e1 -
         g1 * std::log((9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) +
                        std::pow(w, 2))) /
           0.2e1 -
         std::atan(((-w + L) / (-g1 + d)) / 0.2e1) * w / 0.2e1 -
         d * std::log(
               (std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 +
         g1 * std::log(
                (std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 -
         0.3141592654e1 * w;
}

float EfficiencyGEM::getMu1CathodeF2(int n)
{
  float d = mGeometryThickness;
  float L = mGeometryHoleDiameter;
  float w = mGeometryWidth;
  float g1 = mGeometryDistancePrevStage;

  return -0.3e1 / 0.2e1 * L * std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (-g1 + d)) / 0.2e1) +
         L * std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (-g1 + d)) / 0.2e1) * n +
         L * std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (-g1 + d)) / 0.2e1) * n -
         std::atan(((2 * n * L + 2 * n * w + L - w) / (-g1 + d)) / 0.2e1) * n * w -
         std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (-g1 + d)) / 0.2e1) * n * w +
         std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (-g1 + d)) / 0.2e1) * n * w +
         std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (-g1 + d)) / 0.2e1) * n * w -
         L * std::atan(((2 * n * L + 2 * n * w + L - w) / (-g1 + d)) / 0.2e1) * n +
         0.5e1 / 0.2e1 * L * std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (-g1 + d)) / 0.2e1) -
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                        4 * std::pow(n, 2) * std::pow(w, 2) - 20 * std::pow(L, 2) * n - 32 * n * L * w -
                        12 * n * std::pow(w, 2) + 25 * std::pow(L, 2) + 30 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d +
                        4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 +
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                       4 * std::pow(n, 2) * std::pow(w, 2) - 20 * std::pow(L, 2) * n - 32 * n * L * w -
                       12 * n * std::pow(w, 2) + 25 * std::pow(L, 2) + 30 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d +
                       4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 -
         L * std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (-g1 + d)) / 0.2e1) / 0.2e1 +
         std::atan(((2 * n * L + 2 * n * w + L - w) / (-g1 + d)) / 0.2e1) * w / 0.2e1 +
         0.3e1 / 0.2e1 * std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (-g1 + d)) / 0.2e1) * w -
         std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (-g1 + d)) / 0.2e1) * w / 0.2e1 -
         0.3e1 / 0.2e1 * std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (-g1 + d)) / 0.2e1) * w +
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                        4 * std::pow(n, 2) * std::pow(w, 2) - 4 * std::pow(L, 2) * n - 16 * n * L * w -
                        12 * n * std::pow(w, 2) + std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d +
                        4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 -
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                       4 * std::pow(n, 2) * std::pow(w, 2) - 4 * std::pow(L, 2) * n - 16 * n * L * w -
                       12 * n * std::pow(w, 2) + std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d +
                       4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 -
         L * std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (-g1 + d)) / 0.2e1) * n -
         L * std::atan(((2 * n * L + 2 * n * w + L - w) / (-g1 + d)) / 0.2e1) / 0.2e1 -
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                        4 * std::pow(n, 2) * std::pow(w, 2) + 4 * std::pow(L, 2) * n - 4 * n * std::pow(w, 2) +
                        std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) +
                        std::pow(w, 2))) /
           0.2e1 +
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                       4 * std::pow(n, 2) * std::pow(w, 2) + 4 * std::pow(L, 2) * n - 4 * n * std::pow(w, 2) +
                       std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) +
                       std::pow(w, 2))) /
           0.2e1 +
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                        4 * std::pow(n, 2) * std::pow(w, 2) - 12 * std::pow(L, 2) * n - 16 * n * L * w -
                        4 * n * std::pow(w, 2) + 9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d +
                        4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 -
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                       4 * std::pow(n, 2) * std::pow(w, 2) - 12 * std::pow(L, 2) * n - 16 * n * L * w -
                       4 * n * std::pow(w, 2) + 9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d +
                       4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1;
}

float EfficiencyGEM::getMu2Cathode()
{
  float result = 0.0;

  for (int n = 2; n <= mNumberHoles; ++n) {
    result += getMu2CathodeF2(n);
  }

  result += getMu2Cathodef2();

  return result;
}

float EfficiencyGEM::getMu2Cathodef2()
{
  float d = mGeometryThickness;
  float L = mGeometryHoleDiameter;
  float w = mGeometryWidth;
  float g1 = mGeometryDistancePrevStage;

  return -0.3e1 / 0.2e1 * L * std::atan(((3 * L + w) / (d + g1)) / 0.2e1) +
         L * std::atan(((-w + L) / (d + g1)) / 0.2e1) / 0.2e1 -
         std::atan(((3 * L + w) / (d + g1)) / 0.2e1) * w / 0.2e1 +
         d * std::log((9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) +
                       std::pow(w, 2))) /
           0.2e1 +
         g1 * std::log((9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) +
                        std::pow(w, 2))) /
           0.2e1 -
         std::atan(((-w + L) / (d + g1)) / 0.2e1) * w / 0.2e1 -
         d * std::log(
               (std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 -
         g1 * std::log(
                (std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1;
}

float EfficiencyGEM::getMu2CathodeF2(int n)
{
  float d = mGeometryThickness;
  float L = mGeometryHoleDiameter;
  float w = mGeometryWidth;
  float g1 = mGeometryDistancePrevStage;

  return -L * std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (d + g1)) / 0.2e1) * n +
         L * std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (d + g1)) / 0.2e1) * n -
         std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (d + g1)) / 0.2e1) * n * w +
         std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (d + g1)) / 0.2e1) * n * w -
         std::atan(((2 * n * L + 2 * n * w + L - w) / (d + g1)) / 0.2e1) * n * w +
         std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (d + g1)) / 0.2e1) * n * w +
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                       4 * std::pow(n, 2) * std::pow(w, 2) + 4 * std::pow(L, 2) * n - 4 * n * std::pow(w, 2) +
                       std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) +
                       std::pow(w, 2))) /
           0.2e1 +
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                        4 * std::pow(n, 2) * std::pow(w, 2) + 4 * std::pow(L, 2) * n - 4 * n * std::pow(w, 2) +
                        std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) +
                        std::pow(w, 2))) /
           0.2e1 -
         L * std::atan(((2 * n * L + 2 * n * w + L - w) / (d + g1)) / 0.2e1) * n -
         L * std::atan(((2 * n * L + 2 * n * w + L - w) / (d + g1)) / 0.2e1) / 0.2e1 +
         0.3e1 / 0.2e1 * std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (d + g1)) / 0.2e1) * w -
         std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (d + g1)) / 0.2e1) * w / 0.2e1 +
         std::atan(((2 * n * L + 2 * n * w + L - w) / (d + g1)) / 0.2e1) * w / 0.2e1 -
         0.3e1 / 0.2e1 * std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (d + g1)) / 0.2e1) * w +
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                       4 * std::pow(n, 2) * std::pow(w, 2) - 20 * std::pow(L, 2) * n - 32 * n * L * w -
                       12 * n * std::pow(w, 2) + 25 * std::pow(L, 2) + 30 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d +
                       4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 +
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                        4 * std::pow(n, 2) * std::pow(w, 2) - 20 * std::pow(L, 2) * n - 32 * n * L * w -
                        12 * n * std::pow(w, 2) + 25 * std::pow(L, 2) + 30 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d +
                        4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 -
         L * std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (d + g1)) / 0.2e1) / 0.2e1 +
         0.5e1 / 0.2e1 * L * std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (d + g1)) / 0.2e1) -
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                        4 * std::pow(n, 2) * std::pow(w, 2) - 4 * std::pow(L, 2) * n - 16 * n * L * w -
                        12 * n * std::pow(w, 2) + std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d +
                        4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 +
         L * std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (d + g1)) / 0.2e1) * n -
         0.3e1 / 0.2e1 * L * std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (d + g1)) / 0.2e1) -
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                       4 * std::pow(n, 2) * std::pow(w, 2) - 4 * std::pow(L, 2) * n - 16 * n * L * w -
                       12 * n * std::pow(w, 2) + std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d +
                       4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 -
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                       4 * std::pow(n, 2) * std::pow(w, 2) - 12 * std::pow(L, 2) * n - 16 * n * L * w -
                       4 * n * std::pow(w, 2) + 9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d +
                       4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 -
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w +
                        4 * std::pow(n, 2) * std::pow(w, 2) - 12 * std::pow(L, 2) * n - 16 * n * L * w -
                        4 * n * std::pow(w, 2) + 9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d +
                        4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1;
}

float EfficiencyGEM::getLambdaCathode()
{
  float result = 0.0;

  for (int n = 2; n <= mNumberHoles; ++n) {
    result += getLambdaCathodeF2(n);
  }

  result += getLambdaCathodef2();

  return result;
}

float EfficiencyGEM::getLambdaCathodef2()
{
  float d = mGeometryThickness;
  float L = mGeometryHoleDiameter;
  float w = mGeometryWidth;
  float g1 = mGeometryDistancePrevStage;

  return -d * std::log(0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d +
                       0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 +
         g1 * std::log(0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d +
                       0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 +
         std::atan((-w + L) / (-g1 + d) / 0.2e1) * w / 0.2e1 +
         0.3e1 / 0.2e1 * L * std::atan((0.3e1 * L + w) / (d + g1) / 0.2e1) +
         0.3e1 / 0.2e1 * L * std::atan((0.3e1 * L + w) / (-g1 + d) / 0.2e1) -
         d * std::log(0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d +
                      0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         g1 * std::log(0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d +
                       0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         L * std::atan((-w + L) / (d + g1) / 0.2e1) / 0.2e1 +
         std::atan((0.3e1 * L + w) / (d + g1) / 0.2e1) * w / 0.2e1 -
         L * std::atan((-w + L) / (-g1 + d) / 0.2e1) / 0.2e1 +
         d * std::log(std::pow(L, 0.2e1) - 0.2e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d +
                      0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         g1 * std::log(std::pow(L, 0.2e1) - 0.2e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d +
                       0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 +
         std::atan((0.3e1 * L + w) / (-g1 + d) / 0.2e1) * w / 0.2e1 +
         std::atan((-w + L) / (d + g1) / 0.2e1) * w / 0.2e1 +
         d * std::log(std::pow(L, 0.2e1) - 0.2e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d +
                      0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 +
         g1 * std::log(std::pow(L, 0.2e1) - 0.2e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d +
                       0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1;
}

float EfficiencyGEM::getLambdaCathodeF2(int n)
{
  float d = mGeometryThickness;
  float L = mGeometryHoleDiameter;
  float w = mGeometryWidth;
  float g1 = mGeometryDistancePrevStage;

  return -d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                       0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) + 0.4e1 * std::pow(L, 0.2e1) * n -
                       0.4e1 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) - 0.2e1 * L * w +
                       0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         g1 * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                       0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) + 0.4e1 * std::pow(L, 0.2e1) * n -
                       0.4e1 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) - 0.2e1 * L * w +
                       0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         0.3e1 / 0.2e1 * std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.5e1 * L - 0.3e1 * w) / (d + g1) / 0.2e1) * w +
         std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.3e1 * L - w) / (d + g1) / 0.2e1) * w / 0.2e1 -
         std::atan((0.2e1 * n * L + 0.2e1 * n * w + L - w) / (d + g1) / 0.2e1) * w / 0.2e1 +
         0.3e1 / 0.2e1 * std::atan((0.2e1 * n * L + 0.2e1 * n * w - L - 0.3e1 * w) / (d + g1) / 0.2e1) * w -
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                      0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.20e2 * std::pow(L, 0.2e1) * n -
                      0.32e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + 0.25e2 * std::pow(L, 0.2e1) +
                      0.30e2 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) +
                      0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 -
         g1 * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                       0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.20e2 * std::pow(L, 0.2e1) * n -
                       0.32e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + 0.25e2 * std::pow(L, 0.2e1) +
                       0.30e2 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) +
                       0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 +
         g1 * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                       0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.4e1 * std::pow(L, 0.2e1) * n -
                       0.16e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) + 0.6e1 * L * w +
                       0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) +
                       0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 +
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                      0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.4e1 * std::pow(L, 0.2e1) * n -
                      0.16e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) + 0.6e1 * L * w +
                      0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) +
                      0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 +
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                      0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.12e2 * std::pow(L, 0.2e1) * n -
                      0.16e2 * n * L * w - 0.4e1 * n * std::pow(w, 0.2e1) + 0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w +
                      0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 +
         g1 *
           std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                    0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.12e2 * std::pow(L, 0.2e1) * n -
                    0.16e2 * n * L * w - 0.4e1 * n * std::pow(w, 0.2e1) + 0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w +
                    0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 +
         L * std::atan((0.2e1 * n * L + 0.2e1 * n * w + L - w) / (-g1 + d) / 0.2e1) / 0.2e1 +
         0.3e1 / 0.2e1 * L * std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.3e1 * L - w) / (-g1 + d) / 0.2e1) -
         0.5e1 / 0.2e1 * L * std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.5e1 * L - 0.3e1 * w) / (-g1 + d) / 0.2e1) +
         g1 * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                       0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.20e2 * std::pow(L, 0.2e1) * n -
                       0.32e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + 0.25e2 * std::pow(L, 0.2e1) +
                       0.30e2 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) +
                       0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 -
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                      0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.20e2 * std::pow(L, 0.2e1) * n -
                      0.32e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + 0.25e2 * std::pow(L, 0.2e1) +
                      0.30e2 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) +
                      0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 -
         std::atan((0.2e1 * n * L + 0.2e1 * n * w + L - w) / (-g1 + d) / 0.2e1) * w / 0.2e1 -
         0.3e1 / 0.2e1 * std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.5e1 * L - 0.3e1 * w) / (-g1 + d) / 0.2e1) * w +
         std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.3e1 * L - w) / (-g1 + d) / 0.2e1) * w / 0.2e1 +
         0.3e1 / 0.2e1 * std::atan((0.2e1 * n * L + 0.2e1 * n * w - L - 0.3e1 * w) / (-g1 + d) / 0.2e1) * w -
         g1 * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                       0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.4e1 * std::pow(L, 0.2e1) * n -
                       0.16e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) + 0.6e1 * L * w +
                       0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) +
                       0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 +
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                      0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.4e1 * std::pow(L, 0.2e1) * n -
                      0.16e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) + 0.6e1 * L * w +
                      0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) +
                      0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 +
         g1 * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                       0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) + 0.4e1 * std::pow(L, 0.2e1) * n -
                       0.4e1 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) - 0.2e1 * L * w +
                       0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                      0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) + 0.4e1 * std::pow(L, 0.2e1) * n -
                      0.4e1 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) - 0.2e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) -
                      0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         g1 *
           std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                    0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.12e2 * std::pow(L, 0.2e1) * n -
                    0.16e2 * n * L * w - 0.4e1 * n * std::pow(w, 0.2e1) + 0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w +
                    0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 +
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                      0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.12e2 * std::pow(L, 0.2e1) * n -
                      0.16e2 * n * L * w - 0.4e1 * n * std::pow(w, 0.2e1) + 0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w +
                      0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         0.5e1 / 0.2e1 * L * std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.5e1 * L - 0.3e1 * w) / (d + g1) / 0.2e1) +
         L * std::atan((0.2e1 * n * L + 0.2e1 * n * w - L - 0.3e1 * w) / (d + g1) / 0.2e1) / 0.2e1 +
         L * std::atan((0.2e1 * n * L + 0.2e1 * n * w + L - w) / (d + g1) / 0.2e1) / 0.2e1 +
         0.3e1 / 0.2e1 * L * std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.3e1 * L - w) / (d + g1) / 0.2e1) +
         L * std::atan((0.2e1 * n * L + 0.2e1 * n * w - L - 0.3e1 * w) / (-g1 + d) / 0.2e1) / 0.2e1 +
         std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.5e1 * L - 0.3e1 * w) / (-g1 + d) / 0.2e1) * n * w -
         std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.3e1 * L - w) / (d + g1) / 0.2e1) * n * w +
         std::atan((0.2e1 * n * L + 0.2e1 * n * w + L - w) / (-g1 + d) / 0.2e1) * n * w -
         L * std::atan((0.2e1 * n * L + 0.2e1 * n * w - L - 0.3e1 * w) / (-g1 + d) / 0.2e1) * n -
         L * std::atan((0.2e1 * n * L + 0.2e1 * n * w - L - 0.3e1 * w) / (d + g1) / 0.2e1) * n +
         L * std::atan((0.2e1 * n * L + 0.2e1 * n * w + L - w) / (-g1 + d) / 0.2e1) * n +
         L * std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.5e1 * L - 0.3e1 * w) / (-g1 + d) / 0.2e1) * n -
         std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.3e1 * L - w) / (-g1 + d) / 0.2e1) * n * w +
         std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.5e1 * L - 0.3e1 * w) / (d + g1) / 0.2e1) * n * w -
         std::atan((0.2e1 * n * L + 0.2e1 * n * w - L - 0.3e1 * w) / (-g1 + d) / 0.2e1) * n * w -
         L * std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.3e1 * L - w) / (-g1 + d) / 0.2e1) * n -
         std::atan((0.2e1 * n * L + 0.2e1 * n * w - L - 0.3e1 * w) / (d + g1) / 0.2e1) * n * w +
         L * std::atan((0.2e1 * n * L + 0.2e1 * n * w + L - w) / (d + g1) / 0.2e1) * n +
         L * std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.5e1 * L - 0.3e1 * w) / (d + g1) / 0.2e1) * n +
         std::atan((0.2e1 * n * L + 0.2e1 * n * w + L - w) / (d + g1) / 0.2e1) * n * w -
         L * std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.3e1 * L - w) / (d + g1) / 0.2e1) * n;
}
