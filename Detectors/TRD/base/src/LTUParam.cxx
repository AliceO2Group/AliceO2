// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TMath.h"
#include "TRDBase/LTUParam.h"
#include "fairlogger/Logger.h"

using namespace o2::trd;

// definition of geometry constants
std::array<float, 30> LTUParam::mgZrow = {
  301, 177, 53, -57, -181,
  301, 177, 53, -57, -181,
  315, 184, 53, -57, -188,
  329, 191, 53, -57, -195,
  343, 198, 53, -57, -202,
  347, 200, 53, -57, -204
};
std::array<float, 6> LTUParam::mgX = { 300.65, 313.25, 325.85, 338.45, 351.05, 363.65 };
std::array<float, 6> LTUParam::mgTiltingAngle = { -2., 2., -2., 2., -2., 2. };
int LTUParam::mgDyMax = 63;
int LTUParam::mgDyMin = -64;
float LTUParam::mgBinDy = 140e-4;
std::array<float, 6> LTUParam::mgWidthPad = { 0.635, 0.665, 0.695, 0.725, 0.755, 0.785 };
std::array<float, 6> LTUParam::mgLengthInnerPadC1 = { 7.5, 7.5, 8.0, 8.5, 9.0, 9.0 };
std::array<float, 6> LTUParam::mgLengthOuterPadC1 = { 7.5, 7.5, 7.5, 7.5, 7.5, 8.5 };
std::array<float, 6> LTUParam::mgInvX;
std::array<float, 6> LTUParam::mgTiltingAngleTan;
std::array<float, 6> LTUParam::mgInvWidthPad;

float LTUParam::mgLengthInnerPadC0 = 9.0;
float LTUParam::mgLengthOuterPadC0 = 8.0;
float LTUParam::mgScalePad = 256. * 32.;
float LTUParam::mgDriftLength = 3.;

LTUParam::LTUParam() : mMagField(0.),
                       mOmegaTau(0.),
                       mPtMin(0.1),
                       mNtimebins(20 << 5),
                       mScaleQ0(0),
                       mScaleQ1(0),
                       mPidTracklengthCorr(false),
                       mTiltCorr(false),
                       mPidGainCorr(false)
{
  // default constructor
  // These variables are used internally in the class to elliminate divisions.
  // putting them at the top was messy.
  int j = 0;
  std::for_each(mgInvX.begin(), mgInvX.end(), [&j](float& x) { x = 1. / mgX[j]; });
  j = 0;
  std::for_each(mgInvWidthPad.begin(), mgInvWidthPad.end(), [&j](float& x) { x = 1. / mgWidthPad[j]; });
  j = 0;
  std::for_each(mgTiltingAngleTan.begin(), mgTiltingAngleTan.end(), [&j](float& x) { x = std::tan(mgTiltingAngle[j] * M_PI / 180.0); });

  mInvPtMin = 1 / mPtMin;
}

LTUParam::~LTUParam() = default;

int LTUParam::getDyCorrection(int det, int rob, int mcm) const
{
  // calculate the correction of the deflection
  // i.e. Lorentz angle and tilt correction (if active)

  int layer = det % 6;

  float dyTilt = (mgDriftLength * std::tan(mgTiltingAngle[layer] * M_PI / 180.) *
                  getLocalZ(det, rob, mcm) * mgInvX[layer]);

  // calculate Lorentz correction
  float dyCorr = -mOmegaTau * mgDriftLength;

  if (mTiltCorr)
    dyCorr += dyTilt; // add tilt correction

  return (int)TMath::Nint(dyCorr * mgScalePad * mgInvWidthPad[layer]);
}

void LTUParam::getDyRange(int det, int rob, int mcm, int ch,
                          int& dyMinInt, int& dyMaxInt) const
{
  // calculate the deflection range in which tracklets are accepted

  dyMinInt = mgDyMin;
  dyMaxInt = mgDyMax;

  // deflection cut is considered for |B| > 0.1 T only
  if (std::abs(mMagField) < 0.1)
    return;

  float e = 0.30;

  float maxDeflTemp = getPerp(det, rob, mcm, ch) / 2. *             // Sekante/2 (cm)
                      (e * 1e-2 * std::abs(mMagField) * mInvPtMin); // 1/R (1/cm)

  float phi = getPhi(det, rob, mcm, ch);
  if (maxDeflTemp < std::cos(phi)) {
    float maxDeflAngle = std::asin(maxDeflTemp);

    float dyMin = (mgDriftLength *
                   std::tan(phi - maxDeflAngle));

    dyMinInt = int(dyMin / mgBinDy);
    // clipping to allowed range
    if (dyMinInt < mgDyMin)
      dyMinInt = mgDyMin;
    else if (dyMinInt > mgDyMax)
      dyMinInt = mgDyMax;

    float dyMax = (mgDriftLength *
                   std::tan(phi + maxDeflAngle));

    dyMaxInt = int(dyMax / mgBinDy);
    // clipping to allowed range
    if (dyMaxInt > mgDyMax)
      dyMaxInt = mgDyMax;
    else if (dyMaxInt < mgDyMin)
      dyMaxInt = mgDyMin;
  } else if (maxDeflTemp < 0.) {
    // this must not happen
    printf("Inconsistent calculation of sin(alpha): %f\n", maxDeflTemp);
  } else {
    // TRD is not reached at the given pt threshold
    // max range
  }

  if ((dyMaxInt - dyMinInt) <= 0) {
    LOG(info) << "strange dy range: [" << dyMinInt << "," << dyMaxInt << "], using max range now";
    dyMaxInt = mgDyMax;
    dyMinInt = mgDyMin;
  }
}

float LTUParam::getElongation(int det, int rob, int mcm, int ch) const
{
  // calculate the ratio of the distance to the primary vertex and the
  // distance in x-direction for the given ADC channel

  int layer = det % 6;

  float elongation = std::abs(getDist(det, rob, mcm, ch) * mgInvX[layer]);

  // sanity check
  if (elongation < 0.001) {
    elongation = 1.;
  }
  return elongation;
}

void LTUParam::getCorrectionFactors(int det, int rob, int mcm, int ch,
                                    unsigned int& cor0, unsigned int& cor1, float gain) const
{
  // calculate the gain correction factors for the given ADC channel
  float Invgain = 1.0;
  if (mPidGainCorr == true)
    Invgain = 1 / gain;

  if (mPidTracklengthCorr == true) {
    float InvElongationOverGain = 1 / getElongation(det, rob, mcm, ch) * Invgain;
    cor0 = (unsigned int)(mScaleQ0 * InvElongationOverGain);
    cor1 = (unsigned int)(mScaleQ1 * InvElongationOverGain);
  } else {
    cor0 = (unsigned int)(mScaleQ0 * Invgain);
    cor1 = (unsigned int)(mScaleQ1 * Invgain);
  }
}

int LTUParam::getNtimebins() const
{
  // return the number of timebins used

  return mNtimebins;
}

float LTUParam::getX(int det, int /* rob */, int /* mcm */) const
{
  // return the distance to the beam axis in x-direction

  int layer = det % 6;
  return mgX[layer];
}

float LTUParam::getLocalY(int det, int rob, int mcm, int ch) const
{
  // get local y-position (r-phi) w.r.t. the chamber centre

  int layer = det % 6;
  // calculate the pad position as in the TRAP
  float ypos = (-4 + 1 + (rob & 0x1) * 4 + (mcm & 0x3)) * 18 - ch - 0.5; // y position in bins of pad widths
  return ypos * mgWidthPad[layer];
}

float LTUParam::getLocalZ(int det, int rob, int mcm) const
{
  // get local z-position w.r.t. to the chamber boundary

  int stack = (det % 30) / 6;
  int layer = det % 6;
  int row = (rob / 2) * 4 + mcm / 4;

  if (stack == 2) {
    if (row == 0)
      return (mgZrow[layer * 6 + stack] - 0.5 * mgLengthOuterPadC0);
    else if (row == 11)
      return (mgZrow[layer * 6 + stack] - 1.5 * mgLengthOuterPadC0 - (row - 1) * mgLengthInnerPadC0);
    else
      return (mgZrow[layer * 6 + stack] - mgLengthOuterPadC0 - (row - 0.5) * mgLengthInnerPadC0);
  } else {
    if (row == 0)
      return (mgZrow[layer * 6 + stack] - 0.5 * mgLengthOuterPadC1[layer]);
    else if (row == 15)
      return (mgZrow[layer * 6 + stack] - 1.5 * mgLengthOuterPadC1[layer] - (row - 1) * mgLengthInnerPadC1[layer]);
    else
      return (mgZrow[layer * 6 + stack] - mgLengthOuterPadC1[layer] - (row - 0.5) * mgLengthInnerPadC1[layer]);
  }
}

float LTUParam::getPerp(int det, int rob, int mcm, int ch) const
{
  // get transverse distance to the beam axis
  float y;
  float x;
  x = getX(det, rob, mcm);
  y = getLocalY(det, rob, mcm, ch);
  return std::sqrt(y * y + x * x);
}

float LTUParam::getPhi(int det, int rob, int mcm, int ch) const
{
  // calculate the azimuthal angle for the given ADC channel

  return std::atan2(getLocalY(det, rob, mcm, ch), getX(det, rob, mcm));
}

float LTUParam::getDist(int det, int rob, int mcm, int ch) const
{
  // calculate the distance from the origin for the given ADC channel
  float x, y, z;
  x = getX(det, rob, mcm);
  y = getLocalY(det, rob, mcm, ch);
  z = getLocalZ(det, rob, mcm);

  return std::sqrt(y * y + x * x + z * z);
}
