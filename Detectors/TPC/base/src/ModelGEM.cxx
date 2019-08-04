// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ModelGEM.cxx
/// \brief Implementations of the model calculations + simulations of the GEM efficiencies
/// \author Viktor Ratza, University of Bonn, ratza@hiskp.uni-bonn.de

#include <cmath>

#include "TPCBase/ModelGEM.h"

using namespace o2::tpc;

ModelGEM::ModelGEM() :

                       mAbsGainScaling(1.0),
                       mAttachment(0.0),
                       mStackEnergyCalculated(0),

                       mFitElecEffPitch{{140.0f, 200.0f, 280.0f}},
                       mFitElecEffWidth{{2.0f * mFitElecEffPitch[0] - mFitElecEffHoleDiameter, 2.0f * mFitElecEffPitch[1] - mFitElecEffHoleDiameter, 2.0f * mFitElecEffPitch[2] - mFitElecEffHoleDiameter}},

                       mFitElecEffTuneEta1{{6.78972f, 11.5214f, 18.788f}},
                       mFitElecEffTuneEta2{{6.88737f, 8.98337f, 9.90459f}},
                       mFitElecEffTuneDiffusion{{1.30061f, 1.30285f, 1.30125f}},

                       mFitAbsGainConstant{{-1.91668f, -1.95479f, -1.98842f}},
                       mFitAbsGainSlope{{0.0183423f, 0.0185194f, 0.0186415f}},

                       mFitSingleGainF0{{0.450676f, 0.457851f, 0.465322f}},
                       mFitSingleGainU0{{210.0f, 210.0f, 210.0f}},
                       mFitSingleGainQ{{0.10f, 0.10f, 0.10f}}

{

  for (int i = 0; i <= 2; ++i) {
    mParamC1[i] = getParameterC1(i);
    mParamC2[i] = getParameterC2(i);
    mParamC3[i] = getParameterC3(i);
    mParamC4[i] = getParameterC4(i);
    mParamC5[i] = getParameterC5(i);
    mParamC6[i] = getParameterC6();
  }
}

float ModelGEM::getElectronCollectionEfficiency(float elecFieldAbove, float gemPotential, int geom)
{
  // Fit model: CollTop 6.0 (fitted in a range of 0.0 <= ElecFieldRatioAbove <= 0.16)
  float ElecFieldGEM = (0.001 * gemPotential) / (0.0001 * mFitElecEffThickness); //in kV/cm
  float ElecFieldRatioAbove = elecFieldAbove / ElecFieldGEM;

  float eta1 = mFitElecEffTuneEta1[geom] * ElecFieldRatioAbove;
  float eta2 = mFitElecEffFieldBelow / mFitElecEffFieldGEM;

  float paramC7Bar = getParameterC7Bar(eta1, eta2, geom);
  float paramC8Bar = getParameterC8Bar(eta1, eta2, geom);
  float paramC9Bar = getParameterC9Bar(eta1, eta2, geom);

  return 2.0 * Pi * (paramC7Bar + paramC9Bar * eta1 + paramC8Bar * eta2) /
         (mParamC1[geom] + mParamC3[geom] * eta1 + mParamC2[geom] * eta2);
}

float ModelGEM::getElectronExtractionEfficiency(float elecFieldBelow, float gemPotential, int geom)
{
  // Fit model: ExtrBot 6.1 (fitted in a range of 0.0 <= ElecFieldRatioBelow <= 0.16)
  float ElecFieldGEM = (0.001 * gemPotential) / (0.0001 * mFitElecEffThickness); //in kV/cm
  float ElecFieldRatioBelow = elecFieldBelow / ElecFieldGEM;

  float eta1 = mFitElecEffFieldAbove / mFitElecEffFieldGEM;
  float eta2 = mFitElecEffTuneEta2[geom] * ElecFieldRatioBelow;

  float paramC7 = getParameterC7(eta1, eta2, geom);
  float paramC8 = getParameterC8(eta1, eta2, geom);
  float paramC9 = getParameterC9(eta1, eta2, geom);

  return 2.0 * Pi * (paramC7 + paramC8 * eta1 + paramC9 * eta2) /
         (std::pow(mFitElecEffTuneDiffusion[geom], 3.5) * mParamC4[geom] + mParamC5[geom] * eta1 + 1.0 / mFitElecEffTuneDiffusion[geom] * mParamC6[geom] * eta2);
}

float ModelGEM::getAbsoluteGain(float gemPotential, int geom)
{
  //We assume exponential curves (fitted in a range of 200V <= gemPotential <= 400V)
  return mAbsGainScaling * std::exp(mFitAbsGainConstant[geom] + mFitAbsGainSlope[geom] * gemPotential);
}

float ModelGEM::getSingleGainFluctuation(float gemPotential, int geom)
{
  return 0.5 * (mFitSingleGainF0[geom] + 1.0) + (1.0 - mFitSingleGainF0[geom]) / Pi * std::atan(-mFitSingleGainQ[geom] * (gemPotential - mFitSingleGainU0[geom]));
}

void ModelGEM::setStackProperties(const std::array<int, 4>& geometry, const std::array<float, 5>& distance, const std::array<float, 4>& potential, const std::array<float, 5>& electricField)
{
  mGeometry = geometry;
  mDistance = distance;
  mPotential = potential;
  mElectricField = electricField;
}

float ModelGEM::getStackEnergyResolution()
{
  float PrimaryCharges = PhotonEnergy / Wi;

  // Number of electrons which are collected at each GEM stage
  std::array<float, 4> NumElectrons;

  const int NumOfGEMs = NumElectrons.size();

  for (int n = 0; n < NumElectrons.size(); ++n) {
    float Attachment = 1.0 - mAttachment * mDistance[n]; //1.0 - 1/cm * cm => unitless

    if (n == 0) {
      NumElectrons[n] = PrimaryCharges * getElectronCollectionEfficiency(mElectricField[n], mPotential[n], mGeometry[n]) * Attachment;

    } else {
      NumElectrons[n] = NumElectrons[n - 1] * getAbsoluteGain(mPotential[n - 1], mGeometry[n - 1]) * getElectronExtractionEfficiency(mElectricField[n], mPotential[n - 1], mGeometry[n - 1]) * getElectronCollectionEfficiency(mElectricField[n], mPotential[n], mGeometry[n]) * Attachment;
    }
  }

  // Number of charges after extraction from last amplification stage
  float Attachment = 1.0 - mAttachment * mDistance[NumOfGEMs];
  NumElectrons[NumOfGEMs] = NumElectrons[NumOfGEMs - 1] * getAbsoluteGain(mPotential[NumOfGEMs - 1], mGeometry[NumOfGEMs - 1]) * getElectronExtractionEfficiency(mElectricField[NumOfGEMs], mPotential[NumOfGEMs - 1], mGeometry[NumOfGEMs - 1]) * Attachment;

  float SigmaOverMuSquare = 0.0;

  for (int n = 0; n <= NumOfGEMs - 1; ++n) {
    float SingleGainFluctuation = getSingleGainFluctuation(mPotential[n], mGeometry[n]);

    if (n == 0) {
      SigmaOverMuSquare += (Fano + SingleGainFluctuation) / NumElectrons[n];
    } else {
      SigmaOverMuSquare += SingleGainFluctuation / NumElectrons[n];
    }
  }

  mStackEnergyCalculated = 1;
  mStackEffectiveGain = NumElectrons[NumOfGEMs] / PrimaryCharges;

  return std::sqrt(SigmaOverMuSquare);
}

float ModelGEM::getStackEffectiveGain()
{
  if (!mStackEnergyCalculated) {
    getStackEnergyResolution();
  }

  return mStackEffectiveGain;
}

float ModelGEM::getParameterC1(int geom)
{
  return -getLambdaCathode(geom);
}

float ModelGEM::getParameterC2(int geom)
{
  return -getMu2Cathode(geom);
}

float ModelGEM::getParameterC3(int geom)
{
  return -getMu1Cathode(geom);
}

float ModelGEM::getParameterC4(int geom)
{
  return (getMu2Top(-mFitElecEffHoleDiameter / 2.0, mFitElecEffHoleDiameter / 2.0, geom) + Pi * mFitElecEffHoleDiameter);
}

float ModelGEM::getParameterC5(int geom)
{
  return -getMu2Top(-mFitElecEffHoleDiameter / 2.0, mFitElecEffHoleDiameter / 2.0, geom);
}

float ModelGEM::getParameterC6()
{
  return Pi * mFitElecEffHoleDiameter;
}

float ModelGEM::getParameterC7(float eta1, float eta2, int geom)
{
  float intXStart = -(mFitElecEffWidth[geom] + mFitElecEffHoleDiameter) / 4.0;
  float intXEnd = getIntXEndBot(eta1, eta2, geom);

  // Flip g1 <-> g2 for calculation since c7bar(g1->g2) = c7
  flipDistanceNextPrevStage();
  float result = getParameterC7BarFromX(intXStart, intXEnd, geom);
  // Flip back g1<->g2 to initial condition
  flipDistanceNextPrevStage();

  return result;
}

float ModelGEM::getParameterC8(float eta1, float eta2, int geom)
{
  float intXStart = -(mFitElecEffWidth[geom] + mFitElecEffHoleDiameter) / 4.0;
  float intXEnd = getIntXEndBot(eta1, eta2, geom);

  flipDistanceNextPrevStage();
  float result = getParameterC8BarFromX(intXStart, intXEnd, geom);
  flipDistanceNextPrevStage();

  return result;
}

float ModelGEM::getParameterC9(float eta1, float eta2, int geom)
{
  float intXStart = -(mFitElecEffWidth[geom] + mFitElecEffHoleDiameter) / 4.0;
  float intXEnd = getIntXEndBot(eta1, eta2, geom);

  flipDistanceNextPrevStage();
  float result = getParameterC9BarFromX(intXStart, intXEnd, geom);
  flipDistanceNextPrevStage();

  return result;
}

float ModelGEM::getParameterC7Bar(float eta1, float eta2, int geom)
{
  float intXStart = -(mFitElecEffWidth[geom] + mFitElecEffHoleDiameter) / 4.0;
  float intXEnd = getIntXEndTop(eta1, eta2, geom);

  return getParameterC7BarFromX(intXStart, intXEnd, geom);
}

float ModelGEM::getParameterC8Bar(float eta1, float eta2, int geom)
{
  float intXStart = -(mFitElecEffWidth[geom] + mFitElecEffHoleDiameter) / 4.0;
  float intXEnd = getIntXEndTop(eta1, eta2, geom);

  return getParameterC8BarFromX(intXStart, intXEnd, geom);
}

float ModelGEM::getParameterC9Bar(float eta1, float eta2, int geom)
{
  float intXStart = -(mFitElecEffWidth[geom] + mFitElecEffHoleDiameter) / 4.0;
  float intXEnd = getIntXEndTop(eta1, eta2, geom);

  return getParameterC9BarFromX(intXStart, intXEnd, geom);
}

float ModelGEM::getParameterC7BarFromX(float intXStart, float intXEnd, int geom)
{
  return (-1.0 / (2.0 * Pi)) * (getLambdaCathode(geom) + 2.0 * getMu2Top(intXStart, intXEnd, geom));
}

float ModelGEM::getParameterC8BarFromX(float intXStart, float intXEnd, int geom)
{
  return (-1.0 / (2.0 * Pi)) * (getMu2Cathode(geom) - 2.0 * getMu2Top(intXStart, intXEnd, geom));
}

float ModelGEM::getParameterC9BarFromX(float intXStart, float intXEnd, int geom)
{
  return (-1.0 / (2.0 * Pi)) * (getMu1Cathode(geom) + 4.0 * Pi * (intXEnd - intXStart));
}

void ModelGEM::flipDistanceNextPrevStage()
{
  float mFitElecEffDistancePrevStageOld = mFitElecEffDistancePrevStage;
  mFitElecEffDistancePrevStage = mFitElecEffDistanceNextStage;
  mFitElecEffDistanceNextStage = mFitElecEffDistancePrevStageOld;
}

float ModelGEM::getIntXEndBot(float eta1, float eta2, int geom)
{
  float result;

  if (eta2 <= getEta2Kink1(eta1, geom)) {
    result = -(mFitElecEffHoleDiameter + mFitElecEffWidth[geom]) / 4.0;
  } else if (getEta2Kink1(eta1, geom) < eta2 && eta2 < getEta2Kink2(eta1, geom)) {
    result = -mFitElecEffPitch[geom] / 2.0 + sqrt(getHtop2(geom) * (eta1 - 1.0) * (2.0 * Pi * eta2 + getHtop0(geom) * (1.0 - eta1))) / (getHtop2(geom) * (eta1 - 1.0));
  } else {
    result = -mFitElecEffHoleDiameter / 2.0;
  }

  return result;
}

float ModelGEM::getIntXEndTop(float eta1, float eta2, int geom)
{
  float result;

  if (eta1 <= getEta1Kink1(eta2, geom)) {
    result = -(mFitElecEffHoleDiameter + mFitElecEffWidth[geom]) / 4.0;
  } else if (getEta1Kink1(eta2, geom) < eta1 && eta1 < getEta1Kink2(eta2, geom)) {
    result = -mFitElecEffPitch[geom] / 2.0 + sqrt(getHtop2(geom) * (eta2 - 1.0) * (2.0 * Pi * eta1 + getHtop0(geom) * (1.0 - eta2))) / (getHtop2(geom) * (eta2 - 1.0));
  } else {
    result = -mFitElecEffHoleDiameter / 2.0;
  }

  return result;
}

float ModelGEM::getEta1Kink2(float eta2, int geom)
{
  return 1.0 / (2.0 * Pi) *
         ((mFitElecEffHoleDiameter - mFitElecEffPitch[geom]) * (mFitElecEffHoleDiameter - mFitElecEffPitch[geom]) / 4.0 * getHtop2(geom) +
          getHtop0(geom)) *
         (eta2 - 1.0);
}

float ModelGEM::getEta1Kink1(float eta2, int geom)
{
  return -1.0 / (2.0 * Pi) * getHtop0(geom) * (1.0 - eta2);
}

float ModelGEM::getEta2Kink2(float eta1, int geom)
{
  return 1.0 / (2.0 * Pi) *
         ((mFitElecEffHoleDiameter - mFitElecEffPitch[geom]) * (mFitElecEffHoleDiameter - mFitElecEffPitch[geom]) / 4.0 * getHtop2(geom) +
          getHtop0(geom)) *
         (eta1 - 1.0);
}

float ModelGEM::getEta2Kink1(float eta1, int geom)
{
  return -1.0 / (2.0 * Pi) * getHtop0(geom) * (1.0 - eta1);
}

float ModelGEM::getHtop0(int geom)
{
  float result = 0.0;

  for (int n = 2; n <= mFitElecEffNumberHoles; ++n) {
    result += getMu2TopFTaylorTerm0(n, geom);
  }

  result += getMu2TopfTaylorTerm0(geom);

  return result;
}

float ModelGEM::getHtop2(int geom)
{
  float result = 0.0;

  for (int n = 2; n <= mFitElecEffNumberHoles; ++n) {
    result += getMu2TopFTaylorTerm2(n, geom);
  }

  result += getMu2TopfTaylorTerm2(geom);

  return result;
}

float ModelGEM::getMu2TopfTaylorTerm2(int geom)
{
  float d = mFitElecEffThickness;
  float L = mFitElecEffHoleDiameter;
  float w = mFitElecEffWidth[geom];

  return (12 * L + 4 * w) / d / ((9 * std::pow(L, 2)) + (6 * L * w) + 0.16e2 * std::pow(d, 0.2e1) + std::pow(w, 2)) /
           (std::pow(0.3e1 / 0.2e1 * L + w / 0.2e1, 0.2e1) * std::pow(d, -0.2e1) / 0.4e1 + 0.1e1) +
         (4 * L - 4 * w) / d / (std::pow(L, 2) - (2 * L * w) + 0.16e2 * std::pow(d, 0.2e1) + std::pow(w, 2)) /
           (std::pow(L / 0.2e1 - w / 0.2e1, 0.2e1) * std::pow(d, -0.2e1) / 0.4e1 + 0.1e1);
}

float ModelGEM::getMu2TopFTaylorTerm2(int n, int geom)
{
  float d = mFitElecEffThickness;
  float L = mFitElecEffHoleDiameter;
  float w = mFitElecEffWidth[geom];

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

float ModelGEM::getMu2TopFTaylorTerm0(int n, int geom)
{
  float d = mFitElecEffThickness;
  float L = mFitElecEffHoleDiameter;
  float w = mFitElecEffWidth[geom];

  return std::atan((((n - 2) * L) + ((n - 1) * w) - w / 0.2e1 - L / 0.2e1) / d / 0.2e1) +
         std::atan((((n - 2) * L) + ((n - 1) * w) + w / 0.2e1 + L / 0.2e1) / d / 0.2e1) -
         std::atan(((n * L) + ((n - 1) * w) - w / 0.2e1 - L / 0.2e1) / d / 0.2e1) -
         std::atan(((n * L) + ((n - 1) * w) + w / 0.2e1 + L / 0.2e1) / d / 0.2e1);
}

float ModelGEM::getMu2TopfTaylorTerm0(int geom)
{
  float d = mFitElecEffThickness;
  float L = mFitElecEffHoleDiameter;
  float w = mFitElecEffWidth[geom];

  return -std::atan((0.3e1 / 0.2e1 * L + w / 0.2e1) / d / 0.2e1) - std::atan((L / 0.2e1 - w / 0.2e1) / d / 0.2e1);
}

float ModelGEM::getMu2Top(float intXStart, float intXEnd, int geom)
{
  float result = 0.0;

  for (int n = 2; n <= mFitElecEffNumberHoles; ++n) {
    result += getMu2TopF2(n, intXStart, intXEnd, geom);
  }

  result += getMu2Topf2(intXStart, intXEnd);

  return result;
}

float ModelGEM::getMu2Topf2(float intXStart, float intXEnd)
{
  float d = mFitElecEffThickness;
  float L = mFitElecEffHoleDiameter;

  return std::atan(((L - 2 * intXStart) / d) / 0.2e1) * intXStart -
         std::atan(((L - 2 * intXStart) / d) / 0.2e1) * L / 0.2e1 +
         d * std::log((std::pow(L, 2) - 4 * L * intXStart + 4 * std::pow(d, 2) + 4 * intXStart * intXStart)) / 0.2e1 +
         std::atan(((L + 2 * intXStart) / d) / 0.2e1) * intXStart +
         std::atan(((L + 2 * intXStart) / d) / 0.2e1) * L / 0.2e1 -
         d * std::log((std::pow(L, 2) + 4 * L * intXStart + 4 * std::pow(d, 2) + 4 * intXStart * intXStart)) / 0.2e1 -
         std::atan(((-2 * intXEnd + L) / d) / 0.2e1) * intXEnd +
         std::atan(((-2 * intXEnd + L) / d) / 0.2e1) * L / 0.2e1 -
         d * std::log((std::pow(L, 2) - 4 * L * intXEnd + 4 * std::pow(d, 2) + 4 * intXEnd * intXEnd)) / 0.2e1 -
         std::atan(((2 * intXEnd + L) / d) / 0.2e1) * intXEnd - std::atan(((2 * intXEnd + L) / d) / 0.2e1) * L / 0.2e1 +
         d * std::log((std::pow(L, 2) + 4 * L * intXEnd + 4 * std::pow(d, 2) + 4 * intXEnd * intXEnd)) / 0.2e1;
}

float ModelGEM::getMu2TopF2(int n, float intXStart, float intXEnd, int geom)
{
  float d = mFitElecEffThickness;
  float L = mFitElecEffHoleDiameter;
  float w = mFitElecEffWidth[geom];

  return std::atan(((n * L + n * w - w + 2 * intXStart) / d) / 0.2e1) * intXStart -
         std::atan(((n * L + n * w - w + 2 * intXStart) / d) / 0.2e1) * w / 0.2e1 +
         std::atan(((n * L + n * w - w - 2 * intXStart) / d) / 0.2e1) * intXStart +
         std::atan(((n * L + n * w - w - 2 * intXStart) / d) / 0.2e1) * w / 0.2e1 +
         std::atan(((n * L + n * w - 2 * L - w + 2 * intXEnd) / d) / 0.2e1) * intXEnd -
         std::atan(((n * L + n * w - 2 * L - w + 2 * intXEnd) / d) / 0.2e1) * L -
         std::atan(((n * L + n * w - 2 * L - w + 2 * intXEnd) / d) / 0.2e1) * w / 0.2e1 +
         std::atan(((n * L + n * w - 2 * L - w - 2 * intXEnd) / d) / 0.2e1) * intXEnd +
         std::atan(((n * L + n * w - 2 * L - w - 2 * intXEnd) / d) / 0.2e1) * L +
         std::atan(((n * L + n * w - 2 * L - w - 2 * intXEnd) / d) / 0.2e1) * w / 0.2e1 -
         std::atan(((n * L + n * w - w + 2 * intXEnd) / d) / 0.2e1) * intXEnd +
         std::atan(((n * L + n * w - w + 2 * intXEnd) / d) / 0.2e1) * w / 0.2e1 -
         std::atan(((n * L + n * w - w - 2 * intXEnd) / d) / 0.2e1) * intXEnd -
         std::atan(((n * L + n * w - w - 2 * intXEnd) / d) / 0.2e1) * w / 0.2e1 -
         std::atan(((n * L + n * w - 2 * L - w + 2 * intXStart) / d) / 0.2e1) * intXStart +
         std::atan(((n * L + n * w - 2 * L - w + 2 * intXStart) / d) / 0.2e1) * L +
         std::atan(((n * L + n * w - 2 * L - w + 2 * intXStart) / d) / 0.2e1) * w / 0.2e1 -
         std::atan(((n * L + n * w - 2 * L - w - 2 * intXStart) / d) / 0.2e1) * intXStart -
         std::atan(((n * L + n * w - 2 * L - w - 2 * intXStart) / d) / 0.2e1) * L -
         std::atan(((n * L + n * w - 2 * L - w - 2 * intXStart) / d) / 0.2e1) * w / 0.2e1 +
         std::atan(((n * L + n * w - 2 * L - w - 2 * intXStart) / d) / 0.2e1) * n * L / 0.2e1 +
         std::atan(((n * L + n * w - 2 * L - w - 2 * intXStart) / d) / 0.2e1) * n * w / 0.2e1 +
         std::atan(((n * L + n * w - w + 2 * intXStart) / d) / 0.2e1) * n * w / 0.2e1 +
         std::atan(((n * L + n * w - w + 2 * intXStart) / d) / 0.2e1) * n * L / 0.2e1 -
         std::atan(((n * L + n * w - w - 2 * intXStart) / d) / 0.2e1) * n * L / 0.2e1 -
         std::atan(((n * L + n * w - w - 2 * intXStart) / d) / 0.2e1) * n * w / 0.2e1 +
         std::atan(((n * L + n * w - 2 * L - w + 2 * intXEnd) / d) / 0.2e1) * n * L / 0.2e1 +
         std::atan(((n * L + n * w - 2 * L - w + 2 * intXEnd) / d) / 0.2e1) * n * w / 0.2e1 -
         std::atan(((n * L + n * w - 2 * L - w - 2 * intXEnd) / d) / 0.2e1) * n * L / 0.2e1 -
         std::atan(((n * L + n * w - 2 * L - w - 2 * intXEnd) / d) / 0.2e1) * n * w / 0.2e1 -
         std::atan(((n * L + n * w - w + 2 * intXEnd) / d) / 0.2e1) * n * L / 0.2e1 -
         std::atan(((n * L + n * w - w + 2 * intXEnd) / d) / 0.2e1) * n * w / 0.2e1 +
         std::atan(((n * L + n * w - w - 2 * intXEnd) / d) / 0.2e1) * n * L / 0.2e1 +
         std::atan(((n * L + n * w - w - 2 * intXEnd) / d) / 0.2e1) * n * w / 0.2e1 -
         std::atan(((n * L + n * w - 2 * L - w + 2 * intXStart) / d) / 0.2e1) * n * L / 0.2e1 -
         std::atan(((n * L + n * w - 2 * L - w + 2 * intXStart) / d) / 0.2e1) * n * w / 0.2e1 +
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) - 4 * std::pow(L, 2) * n - 6 * n * L * w - 4 * n * L * intXEnd - 2 * n * std::pow(w, 2) - 4 * intXEnd * w * n + 4 * std::pow(L, 2) + 4 * L * w + 8 * L * intXEnd + 4 * std::pow(d, 2) + std::pow(w, 2) + 4 * intXEnd * w + 4 * intXEnd * intXEnd)) /
           0.2e1 -
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) - 2 * n * L * w + 4 * n * L * intXStart - 2 * n * std::pow(w, 2) + 4 * w * intXStart * n + 4 * std::pow(d, 2) + std::pow(w, 2) - 4 * w * intXStart + 4 * intXStart * intXStart)) /
           0.2e1 -
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) - 2 * n * L * w - 4 * n * L * intXEnd - 2 * n * std::pow(w, 2) - 4 * intXEnd * w * n + 4 * std::pow(d, 2) + std::pow(w, 2) + 4 * intXEnd * w + 4 * intXEnd * intXEnd)) /
           0.2e1 -
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) - 4 * std::pow(L, 2) * n - 6 * n * L * w - 4 * n * L * intXStart - 2 * n * std::pow(w, 2) - 4 * w * intXStart * n + 4 * std::pow(L, 2) + 4 * L * w + 8 * L * intXStart + 4 * std::pow(d, 2) + std::pow(w, 2) + 4 * w * intXStart + 4 * intXStart * intXStart)) /
           0.2e1 +
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) - 4 * std::pow(L, 2) * n - 6 * n * L * w + 4 * n * L * intXStart - 2 * n * std::pow(w, 2) + 4 * w * intXStart * n + 4 * std::pow(L, 2) + 4 * L * w - 8 * L * intXStart + 4 * std::pow(d, 2) + std::pow(w, 2) - 4 * w * intXStart + 4 * intXStart * intXStart)) /
           0.2e1 -
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) - 4 * std::pow(L, 2) * n - 6 * n * L * w + 4 * n * L * intXEnd - 2 * n * std::pow(w, 2) + 4 * intXEnd * w * n + 4 * std::pow(L, 2) + 4 * L * w - 8 * L * intXEnd + 4 * std::pow(d, 2) + std::pow(w, 2) - 4 * intXEnd * w + 4 * intXEnd * intXEnd)) /
           0.2e1 +
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) - 2 * n * L * w - 4 * n * L * intXStart - 2 * n * std::pow(w, 2) - 4 * w * intXStart * n + 4 * std::pow(d, 2) + std::pow(w, 2) + 4 * w * intXStart + 4 * intXStart * intXStart)) /
           0.2e1 +
         d * std::log((std::pow(L, 2) * std::pow(n, 2) + 2 * L * std::pow(n, 2) * w + std::pow(n, 2) * std::pow(w, 2) - 2 * n * L * w + 4 * n * L * intXEnd - 2 * n * std::pow(w, 2) + 4 * intXEnd * w * n + 4 * std::pow(d, 2) + std::pow(w, 2) - 4 * intXEnd * w + 4 * intXEnd * intXEnd)) /
           0.2e1;
}

float ModelGEM::getMu1Cathode(int geom)
{
  float result = 0.0;

  for (int n = 2; n <= mFitElecEffNumberHoles; ++n) {
    result += getMu1CathodeF2(n, geom);
  }

  result += getMu1Cathodef2(geom);

  return result;
}

float ModelGEM::getMu1Cathodef2(int geom)
{
  float d = mFitElecEffThickness;
  float L = mFitElecEffHoleDiameter;
  float w = mFitElecEffWidth[geom];
  float g1 = mFitElecEffDistancePrevStage;

  return -0.3e1 / 0.2e1 * L * std::atan(((3 * L + w) / (-g1 + d)) / 0.2e1) +
         L * std::atan(((-w + L) / (-g1 + d)) / 0.2e1) / 0.2e1 - L * 0.3141592654e1 -
         std::atan(((3 * L + w) / (-g1 + d)) / 0.2e1) * w / 0.2e1 +
         d * std::log((9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 -
         g1 * std::log((9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 -
         std::atan(((-w + L) / (-g1 + d)) / 0.2e1) * w / 0.2e1 -
         d * std::log((std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 +
         g1 * std::log((std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 -
         0.3141592654e1 * w;
}

float ModelGEM::getMu1CathodeF2(int n, int geom)
{
  float d = mFitElecEffThickness;
  float L = mFitElecEffHoleDiameter;
  float w = mFitElecEffWidth[geom];
  float g1 = mFitElecEffDistancePrevStage;

  return -0.3e1 / 0.2e1 * L * std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (-g1 + d)) / 0.2e1) +
         L * std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (-g1 + d)) / 0.2e1) * n +
         L * std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (-g1 + d)) / 0.2e1) * n -
         std::atan(((2 * n * L + 2 * n * w + L - w) / (-g1 + d)) / 0.2e1) * n * w -
         std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (-g1 + d)) / 0.2e1) * n * w +
         std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (-g1 + d)) / 0.2e1) * n * w +
         std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (-g1 + d)) / 0.2e1) * n * w -
         L * std::atan(((2 * n * L + 2 * n * w + L - w) / (-g1 + d)) / 0.2e1) * n +
         0.5e1 / 0.2e1 * L * std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (-g1 + d)) / 0.2e1) -
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) - 20 * std::pow(L, 2) * n - 32 * n * L * w - 12 * n * std::pow(w, 2) + 25 * std::pow(L, 2) + 30 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 +
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) - 20 * std::pow(L, 2) * n - 32 * n * L * w - 12 * n * std::pow(w, 2) + 25 * std::pow(L, 2) + 30 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 -
         L * std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (-g1 + d)) / 0.2e1) / 0.2e1 +
         std::atan(((2 * n * L + 2 * n * w + L - w) / (-g1 + d)) / 0.2e1) * w / 0.2e1 +
         0.3e1 / 0.2e1 * std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (-g1 + d)) / 0.2e1) * w -
         std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (-g1 + d)) / 0.2e1) * w / 0.2e1 -
         0.3e1 / 0.2e1 * std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (-g1 + d)) / 0.2e1) * w +
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) - 4 * std::pow(L, 2) * n - 16 * n * L * w - 12 * n * std::pow(w, 2) + std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 -
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) - 4 * std::pow(L, 2) * n - 16 * n * L * w - 12 * n * std::pow(w, 2) + std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 -
         L * std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (-g1 + d)) / 0.2e1) * n -
         L * std::atan(((2 * n * L + 2 * n * w + L - w) / (-g1 + d)) / 0.2e1) / 0.2e1 -
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) + 4 * std::pow(L, 2) * n - 4 * n * std::pow(w, 2) + std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 +
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) + 4 * std::pow(L, 2) * n - 4 * n * std::pow(w, 2) + std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 +
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) - 12 * std::pow(L, 2) * n - 16 * n * L * w - 4 * n * std::pow(w, 2) + 9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 -
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) - 12 * std::pow(L, 2) * n - 16 * n * L * w - 4 * n * std::pow(w, 2) + 9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) - 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1;
}

float ModelGEM::getMu2Cathode(int geom)
{
  float result = 0.0;

  for (int n = 2; n <= mFitElecEffNumberHoles; ++n) {
    result += getMu2CathodeF2(n, geom);
  }

  result += getMu2Cathodef2(geom);

  return result;
}

float ModelGEM::getMu2Cathodef2(int geom)
{
  float d = mFitElecEffThickness;
  float L = mFitElecEffHoleDiameter;
  float w = mFitElecEffWidth[geom];
  float g1 = mFitElecEffDistancePrevStage;

  return -0.3e1 / 0.2e1 * L * std::atan(((3 * L + w) / (d + g1)) / 0.2e1) +
         L * std::atan(((-w + L) / (d + g1)) / 0.2e1) / 0.2e1 -
         std::atan(((3 * L + w) / (d + g1)) / 0.2e1) * w / 0.2e1 +
         d * std::log((9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 +
         g1 * std::log((9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 -
         std::atan(((-w + L) / (d + g1)) / 0.2e1) * w / 0.2e1 -
         d * std::log((std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 -
         g1 * std::log((std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1;
}

float ModelGEM::getMu2CathodeF2(int n, int geom)
{
  float d = mFitElecEffThickness;
  float L = mFitElecEffHoleDiameter;
  float w = mFitElecEffWidth[geom];
  float g1 = mFitElecEffDistancePrevStage;

  return -L * std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (d + g1)) / 0.2e1) * n +
         L * std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (d + g1)) / 0.2e1) * n -
         std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (d + g1)) / 0.2e1) * n * w +
         std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (d + g1)) / 0.2e1) * n * w -
         std::atan(((2 * n * L + 2 * n * w + L - w) / (d + g1)) / 0.2e1) * n * w +
         std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (d + g1)) / 0.2e1) * n * w +
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) + 4 * std::pow(L, 2) * n - 4 * n * std::pow(w, 2) + std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 +
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) + 4 * std::pow(L, 2) * n - 4 * n * std::pow(w, 2) + std::pow(L, 2) - 2 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 -
         L * std::atan(((2 * n * L + 2 * n * w + L - w) / (d + g1)) / 0.2e1) * n -
         L * std::atan(((2 * n * L + 2 * n * w + L - w) / (d + g1)) / 0.2e1) / 0.2e1 +
         0.3e1 / 0.2e1 * std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (d + g1)) / 0.2e1) * w -
         std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (d + g1)) / 0.2e1) * w / 0.2e1 +
         std::atan(((2 * n * L + 2 * n * w + L - w) / (d + g1)) / 0.2e1) * w / 0.2e1 -
         0.3e1 / 0.2e1 * std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (d + g1)) / 0.2e1) * w +
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) - 20 * std::pow(L, 2) * n - 32 * n * L * w - 12 * n * std::pow(w, 2) + 25 * std::pow(L, 2) + 30 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 +
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) - 20 * std::pow(L, 2) * n - 32 * n * L * w - 12 * n * std::pow(w, 2) + 25 * std::pow(L, 2) + 30 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 -
         L * std::atan(((2 * n * L + 2 * n * w - L - 3 * w) / (d + g1)) / 0.2e1) / 0.2e1 +
         0.5e1 / 0.2e1 * L * std::atan(((2 * n * L + 2 * n * w - 5 * L - 3 * w) / (d + g1)) / 0.2e1) -
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) - 4 * std::pow(L, 2) * n - 16 * n * L * w - 12 * n * std::pow(w, 2) + std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 +
         L * std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (d + g1)) / 0.2e1) * n -
         0.3e1 / 0.2e1 * L * std::atan(((2 * n * L + 2 * n * w - 3 * L - w) / (d + g1)) / 0.2e1) -
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) - 4 * std::pow(L, 2) * n - 16 * n * L * w - 12 * n * std::pow(w, 2) + std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + 9 * std::pow(w, 2))) /
           0.2e1 -
         d * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) - 12 * std::pow(L, 2) * n - 16 * n * L * w - 4 * n * std::pow(w, 2) + 9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1 -
         g1 * std::log((4 * std::pow(L, 2) * std::pow(n, 2) + 8 * L * std::pow(n, 2) * w + 4 * std::pow(n, 2) * std::pow(w, 2) - 12 * std::pow(L, 2) * n - 16 * n * L * w - 4 * n * std::pow(w, 2) + 9 * std::pow(L, 2) + 6 * L * w + 4 * std::pow(d, 2) + 8 * g1 * d + 4 * std::pow(g1, 2) + std::pow(w, 2))) /
           0.2e1;
}

float ModelGEM::getLambdaCathode(int geom)
{
  float result = 0.0;

  for (int n = 2; n <= mFitElecEffNumberHoles; ++n) {
    result += getLambdaCathodeF2(n, geom);
  }

  result += getLambdaCathodef2(geom);

  return result;
}

float ModelGEM::getLambdaCathodef2(int geom)
{
  float d = mFitElecEffThickness;
  float L = mFitElecEffHoleDiameter;
  float w = mFitElecEffWidth[geom];
  float g1 = mFitElecEffDistancePrevStage;

  return -d * std::log(0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 +
         g1 * std::log(0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 +
         std::atan((-w + L) / (-g1 + d) / 0.2e1) * w / 0.2e1 +
         0.3e1 / 0.2e1 * L * std::atan((0.3e1 * L + w) / (d + g1) / 0.2e1) +
         0.3e1 / 0.2e1 * L * std::atan((0.3e1 * L + w) / (-g1 + d) / 0.2e1) -
         d * std::log(0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         g1 * std::log(0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         L * std::atan((-w + L) / (d + g1) / 0.2e1) / 0.2e1 +
         std::atan((0.3e1 * L + w) / (d + g1) / 0.2e1) * w / 0.2e1 -
         L * std::atan((-w + L) / (-g1 + d) / 0.2e1) / 0.2e1 +
         d * std::log(std::pow(L, 0.2e1) - 0.2e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         g1 * std::log(std::pow(L, 0.2e1) - 0.2e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 +
         std::atan((0.3e1 * L + w) / (-g1 + d) / 0.2e1) * w / 0.2e1 +
         std::atan((-w + L) / (d + g1) / 0.2e1) * w / 0.2e1 +
         d * std::log(std::pow(L, 0.2e1) - 0.2e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 +
         g1 * std::log(std::pow(L, 0.2e1) - 0.2e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1;
}

float ModelGEM::getLambdaCathodeF2(int n, int geom)
{
  float d = mFitElecEffThickness;
  float L = mFitElecEffHoleDiameter;
  float w = mFitElecEffWidth[geom];
  float g1 = mFitElecEffDistancePrevStage;

  return -d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) + 0.4e1 * std::pow(L, 0.2e1) * n - 0.4e1 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) - 0.2e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         g1 * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) + 0.4e1 * std::pow(L, 0.2e1) * n - 0.4e1 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) - 0.2e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         0.3e1 / 0.2e1 * std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.5e1 * L - 0.3e1 * w) / (d + g1) / 0.2e1) * w +
         std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.3e1 * L - w) / (d + g1) / 0.2e1) * w / 0.2e1 -
         std::atan((0.2e1 * n * L + 0.2e1 * n * w + L - w) / (d + g1) / 0.2e1) * w / 0.2e1 +
         0.3e1 / 0.2e1 * std::atan((0.2e1 * n * L + 0.2e1 * n * w - L - 0.3e1 * w) / (d + g1) / 0.2e1) * w -
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.20e2 * std::pow(L, 0.2e1) * n - 0.32e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + 0.25e2 * std::pow(L, 0.2e1) + 0.30e2 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + 0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 -
         g1 * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.20e2 * std::pow(L, 0.2e1) * n - 0.32e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + 0.25e2 * std::pow(L, 0.2e1) + 0.30e2 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + 0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 +
         g1 * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.4e1 * std::pow(L, 0.2e1) * n - 0.16e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + 0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 +
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.4e1 * std::pow(L, 0.2e1) * n - 0.16e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + 0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 +
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.12e2 * std::pow(L, 0.2e1) * n - 0.16e2 * n * L * w - 0.4e1 * n * std::pow(w, 0.2e1) + 0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) + 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
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
         g1 * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.20e2 * std::pow(L, 0.2e1) * n - 0.32e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + 0.25e2 * std::pow(L, 0.2e1) + 0.30e2 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + 0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 -
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.20e2 * std::pow(L, 0.2e1) * n - 0.32e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + 0.25e2 * std::pow(L, 0.2e1) + 0.30e2 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + 0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 -
         std::atan((0.2e1 * n * L + 0.2e1 * n * w + L - w) / (-g1 + d) / 0.2e1) * w / 0.2e1 -
         0.3e1 / 0.2e1 * std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.5e1 * L - 0.3e1 * w) / (-g1 + d) / 0.2e1) * w +
         std::atan((0.2e1 * n * L + 0.2e1 * n * w - 0.3e1 * L - w) / (-g1 + d) / 0.2e1) * w / 0.2e1 +
         0.3e1 / 0.2e1 * std::atan((0.2e1 * n * L + 0.2e1 * n * w - L - 0.3e1 * w) / (-g1 + d) / 0.2e1) * w -
         g1 * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.4e1 * std::pow(L, 0.2e1) * n - 0.16e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + 0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 +
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.4e1 * std::pow(L, 0.2e1) * n - 0.16e2 * n * L * w - 0.12e2 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + 0.9e1 * std::pow(w, 0.2e1)) /
           0.2e1 +
         g1 * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) + 0.4e1 * std::pow(L, 0.2e1) * n - 0.4e1 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) - 0.2e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) + 0.4e1 * std::pow(L, 0.2e1) * n - 0.4e1 * n * std::pow(w, 0.2e1) + std::pow(L, 0.2e1) - 0.2e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 -
         g1 *
           std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w +
                    0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.12e2 * std::pow(L, 0.2e1) * n -
                    0.16e2 * n * L * w - 0.4e1 * n * std::pow(w, 0.2e1) + 0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w +
                    0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
           0.2e1 +
         d * std::log(0.4e1 * std::pow(L, 0.2e1) * std::pow(n, 0.2e1) + 0.8e1 * L * std::pow(n, 0.2e1) * w + 0.4e1 * std::pow(n, 0.2e1) * std::pow(w, 0.2e1) - 0.12e2 * std::pow(L, 0.2e1) * n - 0.16e2 * n * L * w - 0.4e1 * n * std::pow(w, 0.2e1) + 0.9e1 * std::pow(L, 0.2e1) + 0.6e1 * L * w + 0.4e1 * std::pow(d, 0.2e1) - 0.8e1 * g1 * d + 0.4e1 * std::pow(g1, 0.2e1) + std::pow(w, 0.2e1)) /
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
