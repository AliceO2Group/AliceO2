// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// Class containing constant simulation parameters                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TRDSimulation/SimParam.h"
#include "TRDSimulation/TRDSimParams.h"
#include "Field/MagneticField.h"
#include <TGeoGlobalMagField.h>
#include <fairlogger/Logger.h>

using namespace o2::trd;

SimParam::SimParam()
{
  mGasMixture = TRDSimParams::Instance().gas;
  mTRF = TRDSimParams::Instance().trf;
  mMu = TRDSimParams::Instance().trf_landau_mu;
  mSigma = TRDSimParams::Instance().trf_landau_sigma;
  init();
}

void SimParam::cacheMagField()
{
  // The magnetic field strength
  const o2::field::MagneticField* fld = static_cast<o2::field::MagneticField*>(TGeoGlobalMagField::Instance()->GetField());
  if (!fld) {
    LOG(fatal) << "Magnetic field is not initialized!";
    return;
  }
  mField = 0.1 * fld->solenoidField(); // kGauss -> Tesla
  mFieldCached = true;
}

float SimParam::getCachedField() const
{
  if (mFieldCached) {
    return mField;
  } else {
    LOG(fatal) << "Magnetic field has not been cached yet";
  }
  return 0.f;
}

double SimParam::timeResponse(double time) const
{
  //
  // Applies the preamp shaper time response
  // (We assume a signal rise time of 0.2us = fTRFlo/2.
  //

  double rt = (time - .5 * mTRFlo) * mInvTRFwid;
  int iBin = (int)rt;
  double dt = rt - iBin;
  if ((iBin >= 0) && (iBin + 1 < mTRFbin)) {
    return mTRFsmp[iBin] + (mTRFsmp[iBin + 1] - mTRFsmp[iBin]) * dt;
  } else {
    return 0.0;
  }
}

double SimParam::crossTalk(double time) const
{
  //
  // Applies the pad-pad capacitive cross talk
  //

  double rt = (time - mTRFlo) * mInvTRFwid;
  int iBin = (int)rt;
  double dt = rt - iBin;
  if ((iBin >= 0) && (iBin + 1 < mTRFbin)) {
    return mCTsmp[iBin] + (mCTsmp[iBin + 1] - mCTsmp[iBin]) * dt;
  } else {
    return 0.0;
  }
}

void SimParam::init()
{
  //
  // initializes the parameter class for a given gas mixture
  //

  if (isXenon()) {
    // The range and the binwidth for the sampled TRF
    mTRFbin = 200;
    // Start 0.2 mus before the signal
    mTRFlo = -0.4;
    // End the maximum drift time after the signal
    mTRFhi = 3.58;
    // Standard gas gain
    mGasGain = 4000.0;
  } else if (isArgon()) {
    // The range and the binwidth for the sampled TRF
    mTRFbin = 50;
    // Start 0.2 mus before the signal
    mTRFlo = 0.02;
    // End the maximum drift time after the signal
    mTRFhi = 1.98;
    // Higher gas gain
    mGasGain = 8000.0;
  } else {
    LOG(fatal) << "Not a valid gas mixture!\n";
  }
  mInvTRFwid = ((float)mTRFbin) / (mTRFhi - mTRFlo); // Inverse of the bin width of the integrated TRF

  // Create the sampled TRF
  sampleTRF();
}

//_____________________________________________________________________________
void SimParam::sampleTRF()
{
  //
  // Samples the new time response function.
  //

  int ipasa = 0;

  // Xenon
  // From Antons measurements with Fe55 source, adjusted by C. Lippmann.
  // time bins are -0.4, -0.38, -0.36, ...., 3.54, 3.56, 3.58 microseconds
  const int kNpasa = 200; // kNpasa should be equal to fTRFbin!
  float xtalk[kNpasa];

  float signals[4][kNpasa] =
    {{0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0002, 0.0007, 0.0026, 0.0089, 0.0253, 0.0612, 0.1319,
      0.2416, 0.3913, 0.5609, 0.7295, 0.8662, 0.9581, 1.0000, 0.9990, 0.9611, 0.8995, 0.8269, 0.7495, 0.6714, 0.5987,
      0.5334, 0.4756, 0.4249, 0.3811, 0.3433, 0.3110, 0.2837, 0.2607, 0.2409, 0.2243, 0.2099, 0.1974, 0.1868, 0.1776,
      0.1695, 0.1627, 0.1566, 0.1509, 0.1457, 0.1407, 0.1362, 0.1317, 0.1274, 0.1233, 0.1196, 0.1162, 0.1131, 0.1102,
      0.1075, 0.1051, 0.1026, 0.1004, 0.0979, 0.0956, 0.0934, 0.0912, 0.0892, 0.0875, 0.0858, 0.0843, 0.0829, 0.0815,
      0.0799, 0.0786, 0.0772, 0.0757, 0.0741, 0.0729, 0.0718, 0.0706, 0.0692, 0.0680, 0.0669, 0.0655, 0.0643, 0.0630,
      0.0618, 0.0607, 0.0596, 0.0587, 0.0576, 0.0568, 0.0558, 0.0550, 0.0541, 0.0531, 0.0522, 0.0513, 0.0505, 0.0497,
      0.0490, 0.0484, 0.0474, 0.0465, 0.0457, 0.0449, 0.0441, 0.0433, 0.0425, 0.0417, 0.0410, 0.0402, 0.0395, 0.0388,
      0.0381, 0.0374, 0.0368, 0.0361, 0.0354, 0.0348, 0.0342, 0.0336, 0.0330, 0.0324, 0.0318, 0.0312, 0.0306, 0.0301,
      0.0296, 0.0290, 0.0285, 0.0280, 0.0275, 0.0270, 0.0265, 0.0260, 0.0256, 0.0251, 0.0246, 0.0242, 0.0238, 0.0233,
      0.0229, 0.0225, 0.0221, 0.0217, 0.0213, 0.0209, 0.0206, 0.0202, 0.0198, 0.0195, 0.0191, 0.0188, 0.0184, 0.0181,
      0.0178, 0.0175, 0.0171, 0.0168, 0.0165, 0.0162, 0.0159, 0.0157, 0.0154, 0.0151, 0.0148, 0.0146, 0.0143, 0.0140,
      0.0138, 0.0135, 0.0133, 0.0131, 0.0128, 0.0126, 0.0124, 0.0121, 0.0119, 0.0120, 0.0115, 0.0113, 0.0111, 0.0109,
      0.0107, 0.0105, 0.0103, 0.0101, 0.0100, 0.0098, 0.0096, 0.0094, 0.0092, 0.0091, 0.0089, 0.0088, 0.0086, 0.0084,
      0.0083, 0.0081, 0.0080, 0.0078}, // Default TRF
     {
       0.00334448, 0.00334448, 0.00334448, 0.00334448, 0.00334448,
       0.00334448, 0.00167224, 0.00167224, 0.00167224, 0.00167224,
       0.00167224, 0.00167224, 0.00167224, 0.00167224, 0.00501672,
       0.0735786, 0.25585284, 0.5367893, 0.7909699, 0.94816054,
       1., 0.96989967, 0.88628763, 0.77424749, 0.67056856,
       0.56856187, 0.48494983, 0.40468227, 0.34448161, 0.2993311,
       0.26086957, 0.22909699, 0.20568562, 0.18561873, 0.17056856,
       0.15719064, 0.14548495, 0.13545151, 0.1270903, 0.12040134,
       0.11204013, 0.10702341, 0.10200669, 0.09698997, 0.09197324,
       0.08862876, 0.08528428, 0.0819398, 0.07859532, 0.07525084,
       0.07190635, 0.07023411, 0.06688963, 0.06521739, 0.06354515,
       0.06187291, 0.05852843, 0.05685619, 0.05518395, 0.05351171,
       0.05183946, 0.05016722, 0.04849498, 0.04682274, 0.0451505,
       0.04347826, 0.04347826, 0.04347826, 0.04180602, 0.04013378,
       0.03846154, 0.03846154, 0.0367893, 0.03511706, 0.03511706,
       0.03344482, 0.03344482, 0.03344482, 0.03177258, 0.03177258,
       0.03177258, 0.03010033, 0.03010033, 0.02842809, 0.02842809,
       0.02842809, 0.02675585, 0.02675585, 0.02508361, 0.02508361,
       0.02675585, 0.02508361, 0.02341137, 0.02341137, 0.02341137,
       0.02341137, 0.02341137, 0.02341137, 0.02173913, 0.02173913,
       0.02173913, 0.02173913, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689,
       0.02006689, 0.02006689, 0.02006689, 0.02006689, 0.02006689}, // TDR TRF
     {
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}, // Delta distribution/no TRF
     {
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}}; // Landau TRF to be filled later
  for (int i = 0; i < 200; i++) {
    float tb = (i * 4. / 200.) - 0.4;
    signals[3][i] = TMath::Landau(tb, mMu, mSigma) / TMath::Landau(mMu - 0.22278 * mSigma, mMu, mSigma); // Fill Landau TRF (and normalize maximum to 1)
  }

  float signal[kNpasa];
  for (int i = 0; i < 200; i++) {
    signal[i] = signals[mTRF][i];
  }
  // With undershoot, positive peak corresponds to ~3% of the main signal:
  for (ipasa = 3; ipasa < kNpasa; ipasa++) {
    xtalk[ipasa] = 0.2 * (signal[ipasa - 2] - signal[ipasa - 3]);
  }
  xtalk[0] = 0.0;
  xtalk[1] = 0.0;
  xtalk[2] = 0.0;

  // Argon
  // Ar measurement with Fe55 source by Anton
  // time bins are 0.02, 0.06, 0.10, ...., 1.90, 1.94, 1.98 microseconds
  const int kNpasaAr = 50;
  float xtalkAr[kNpasaAr];
  float signalAr[kNpasaAr] = {-0.01, 0.01, 0.00, 0.00, 0.01, -0.01, 0.01, 2.15, 22.28, 55.53, 68.52, 58.21, 40.92,
                              27.12, 18.49, 13.42, 10.48, 8.67, 7.49, 6.55, 5.71, 5.12, 4.63, 4.22, 3.81, 3.48,
                              3.20, 2.94, 2.77, 2.63, 2.50, 2.37, 2.23, 2.13, 2.03, 1.91, 1.83, 1.75, 1.68,
                              1.63, 1.56, 1.49, 1.50, 1.49, 1.29, 1.19, 1.21, 1.21, 1.20, 1.10};
  // Normalization to maximum
  for (ipasa = 0; ipasa < kNpasaAr; ipasa++) {
    signalAr[ipasa] /= 68.52;
  }
  signalAr[0] = 0.0;
  signalAr[1] = 0.0;
  signalAr[2] = 0.0;
  // With undershoot, positive peak corresponds to ~3% of the main signal:
  for (ipasa = 3; ipasa < kNpasaAr; ipasa++) {
    xtalkAr[ipasa] = 0.2 * (signalAr[ipasa - 2] - signalAr[ipasa - 3]);
  }
  xtalkAr[0] = 0.0;
  xtalkAr[1] = 0.0;
  xtalkAr[2] = 0.0;

  for (int iBin = 0; iBin < mTRFbin; iBin++) {
    if (isXenon()) {
      mTRFsmp[iBin] = signal[iBin];
      mCTsmp[iBin] = xtalk[iBin];
    } else if (isArgon()) {
      mTRFsmp[iBin] = signalAr[iBin];
      mCTsmp[iBin] = xtalkAr[iBin];
    }
  }
}
