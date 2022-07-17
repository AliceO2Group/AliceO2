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

/// \file CaloRawFitterGS.cxx
/// \author Dmitri Peresunko

#include <gsl/span>

#include "PHOSReconstruction/CaloRawFitterGS.h"
#include "PHOSBase/PHOSSimParams.h"

using namespace o2::phos;
CaloRawFitterGS::CaloRawFitterGS() : CaloRawFitter()
{
  mDecTime = o2::phos::PHOSSimParams::Instance().mSampleDecayTime;
  mQAccuracy = o2::phos::PHOSSimParams::Instance().mSampleTimeFitAccuracy;
  init();
}

void CaloRawFitterGS::init()
{
  mSpikeThreshold = 5;
  // prepare fitting arrays, once per lifetime
  double k = o2::phos::PHOSSimParams::Instance().mSampleDecayTime;
  ma0[0] = 0.;
  ma1[0] = 0.;
  ma2[0] = 0.;
  ma3[0] = 0.;
  ma4[0] = 0.;
  for (int i = 1; i < NMAXSAMPLES; i++) {
    double xi = k * i;
    mexp[i] = exp(-xi);
    double s = mexp[i] * mexp[i];
    ma0[i] = ma0[i - 1] + s;
    ma1[i] = ma1[i - 1] + s * xi;
    ma2[i] = ma2[i - 1] + s * xi * xi;
    ma3[i] = ma3[i - 1] + s * xi * xi * xi;
    ma4[i] = ma4[i - 1] + s * xi * xi * xi * xi;
  }
}

CaloRawFitterGS::FitStatus CaloRawFitterGS::evaluate(gsl::span<short unsigned int> signal)
{

  // Pedestal analysis mode
  if (mPedestalRun) {
    int nPed = signal.size();
    mAmp = 0.;
    mTime = 0.;
    for (auto a : signal) {
      mAmp += a;
      mTime += a * a;
    }
    if (nPed > 0) {
      mAmp /= nPed;
      mTime = mTime / nPed - mAmp * mAmp;
      if (mTime > 0.) {
        mTime = sqrt(mTime);
      }
    }
    mOverflow = false;
    mStatus = kOK;
    return kOK;
  }

  mStatus = kNotEvaluated;
  // Extract amplitude and time
  mStatus = evalFit(signal);
  return mStatus;
}

CaloRawFitterGS::FitStatus CaloRawFitterGS::evalFit(gsl::span<short unsigned int> signal)
{
  // Calculate signal parameters (energy, time, quality) from array of samples
  // Fit with semi-gaus function with free parameters time and amplitude
  // Signal overflows if there are at least 3 samples of the same amplitude above 900

  // Calculate signal parameters (energy, time, quality) from array of samples
  // Energy is a maximum sample minus pedestal 9
  // Time is the first time bin
  // Signal overflows is there are at least 3 samples of the same amplitude above 900

  int nSamples = signal.size();
  if (nSamples == 0) {
    mAmp = 0;
    mTime = 0.;
    mChi2 = 0.;
    return kEmptyBunch;
  }
  if (nSamples == 1) {
    mAmp = signal[0];
    mTime = 0.;
    mChi2 = 1.;
    return kOK;
  }

  mOverflow = false;

  // if pedestal should be subtracted first evaluate it
  float pedMean = 0;
  int nPed = 0;
  if (mPedSubtract) {
    // remember inverse time order
    for (auto it = signal.rbegin(); (nPed < mPreSamples) && it != signal.rend(); ++it) {
      nPed++;
      pedMean += *it;
    }
    if (nPed > 0) {
      pedMean /= nPed;
    }
    nSamples -= mPreSamples;
    if (nSamples <= 0) { // empty bunch left
      mAmp = 0;
      mTime = 0.;
      mChi2 = 0.;
      return kEmptyBunch;
    }
  }

  float maxSample = 0.;                                    // maximal sample value
  int nMax = 0;                                            // number of consequitive maximal samples
  bool spike = false;                                      // spike in previoud signal bin?
  short ap = -1, app = -1;                                 // remember previous values to evaluate spikes
  double b0 = 0., b1 = 0., b2 = 0., y2 = 0.;               // fit coeficients
  double sa0 = 0., sa1 = 0., sa2 = 0., sa3 = 0., sa4 = 0.; // corrections in case of overflow

  int firstS = nSamples;
  int j = TMath::Min(nSamples, NMAXSAMPLES - 1);
  for (int i = 1; i <= j; i++) {
    short a = signal[firstS - i] - pedMean; // remember inverse order of samples
    float xi = i * mDecTime;
    if (a > maxSample) {
      maxSample = a;
      nMax = 1;
    } else {
      if (a == maxSample) {
        nMax++;
      }
    }
    // check if there was a spike in previous step?
    if (app > 0 && ap > 0) {
      spike = (ap - a > mSpikeThreshold) && (ap - app > mSpikeThreshold);
    }
    if (spike) { // Try to recover: subtract last point contribution and replace by average of "app" and "a"
      float atmp = 0.5 * (app + a);
      float xiprev = xi - mDecTime;
      float st = (atmp - ap) * mexp[i - 1];
      b0 += st;
      b1 += st * xiprev;
      b2 += st * xiprev * xiprev;
      y2 += atmp * atmp - ap * ap;
      ap = a;
    } else {
      app = ap;
      ap = a;
    }
    // Check if in saturation
    if (maxSample > 900 && nMax >= 3) {
      // Remove overflow points from the fit
      if (!mOverflow) {            // first time in this sample: remove two previous points
        sa0 = ma0[i] - ma0[i - 2]; // can not appear at i<2
        sa1 = ma1[i] - ma1[i - 2];
        sa2 = ma2[i] - ma2[i - 2];
        sa3 = ma3[i] - ma3[i - 2];
        sa4 = ma4[i] - ma4[i - 2];
        float st = ap * mexp[i - 1];
        float xiprev = xi - mDecTime;
        b0 -= st;
        b1 -= st * xiprev;
        b2 -= st * xiprev * xiprev;
        y2 -= ap * ap;
        st = app * mexp[i - 2];
        xiprev -= mDecTime;
        b0 -= st;
        b1 -= st * xiprev;
        b2 -= st * xiprev * xiprev;
        y2 -= ap * ap;
      }
      mOverflow = true;
    }
    if (!mOverflow) {
      // to calculate time
      float st = a * mexp[i];
      b0 += st;
      b1 += st * xi;
      b2 += st * xi * xi;
      y2 += a * a;
    } else {                     // do not add current point and subtract contributions to amx[] arrays
      sa0 = ma0[i] - ma0[i - 1]; // can not appear at i<2
      sa1 = ma1[i] - ma1[i - 1];
      sa2 = ma2[i] - ma2[i - 1];
      sa3 = ma3[i] - ma3[i - 1];
      sa4 = ma4[i] - ma4[i - 1];
    }
  } // Scanned full

  // too small amplitude, assing max to max Amp and time to zero and do not calculate height
  if (maxSample < mMinTimeCalc) {
    mAmp = maxSample;
    mTime = 0.;
    mChi2 = 0.;
    return kOK;
  }

  if (mOverflow && b0 == 0) { // strong overflow, no reasonable counts, can not extract anything
    mAmp = 0.;
    mTime = 0.;
    mChi2 = 900.;
    return kOverflow;
  }

  // calculate time, amp and chi2
  double a, b, c, d, e; // Polinomial coefficients
  if (!mOverflow) {
    a = ma1[j] * b0 - ma0[j] * b1;
    b = ma0[j] * b2 + 2. * ma1[j] * b1 - 3. * ma2[j] * b0;
    c = 3. * (ma3[j] * b0 - ma1[j] * b2);
    d = 3. * ma2[j] * b2 - ma4[j] * b0 - 2. * ma3[j] * b1;
    e = ma4[j] * b1 - ma3[j] * b2;
  } else { // account removed points in overflow
    a = (ma1[j] - sa1) * b0 - (ma0[j] - sa0) * b1;
    b = (ma0[j] - sa0) * b2 + 2. * (ma1[j] - sa1) * b1 - 3. * (ma2[j] - sa2) * b0;
    c = 3. * ((ma3[j] - sa3) * b0 - (ma1[j] - sa1) * b2);
    d = 3. * (ma2[j] - sa2) * b2 - (ma4[j] - sa4) * b0 - 2. * (ma3[j] - sa4) * b1;
    e = (ma4[j] - sa4) * b1 - (ma3[j] - sa3) * b2;
  }

  // Find zero of 4-order polinomial
  // first use linear extrapolation to reach correct root of four
  double z = -1.;
  if (ma0[j] * b1 - ma1[j] * b0 != 0) {
    z = (ma1[j] * b1 - ma2[j] * b0) / (ma0[j] * b1 - ma1[j] * b0) - 1.; // linear fit + offset
  }
  double q = 0., dq = 0., ddq = 0., lq = 0., dz = 0.1;
  double z2 = z * z;
  double z3 = z2 * z;
  double z4 = z2 * z2;
  q = a * z4 + b * z3 + c * z2 + d * z + e;        // polinomial
  dq = 4. * a * z3 + 3. * b * z2 + 2. * c * z + d; // Derivative
  ddq = 12. * a * z2 + 6. * b * z + 2. * c;        // Second derivative
  if (dq != 0.) {
    lq = q * ddq / (dq * dq);
  }
  // dz = -q/dq ;               // Newton  ~7 terations
  // dz =-(1+0.5*lq)*q/dq ;     // Chebyshev ~3 iterations to reach |q|<1.e-11
  double ttt = dq * (1. - 0.5 * lq); // Halley’s method ~3 iterations, a bit more precise
  if (ttt != 0) {
    dz = -q / ttt;
  } else {
    dz = 0.1; // step off saddle point
  }
  int it = 0;
  while (TMath::Abs(q) > 0.0001 && (++it < 15)) {
    z += dz;
    z2 = z * z;
    z3 = z2 * z;
    z4 = z2 * z2;
    q = a * z4 + b * z3 + c * z2 + d * z + e;
    dq = 4. * a * z3 + 3. * b * z2 + 2. * c * z + d;
    ddq = 12. * a * z2 + 6. * b * z + 2. * c;
    if (dq != 0) {
      lq = q * ddq / (dq * dq);
      ttt = dq * (1. - 0.5 * lq);
      // dz = -q/dq ;  //Newton
      // dz =-(1+0.5*lq)*q/dq ; //Chebyshev
      if (ttt != 0) {
        dz = -q / ttt; // Halley’s
      } else {
        dz = -q / dq;
      }
    } else {
      dz = 0.5 * dz; // step off saddle point
    }
  }

  // check that result is reasonable
  double denom = ma4[j] - 4. * ma3[j] * z + 6. * ma2[j] * z * z - 4. * ma1[j] * z * z * z + ma0[j] * z * z * z * z;
  if (denom != 0.) {
    mAmp = 4. * exp(-2 - z) * (b2 - 2. * b1 * z + b0 * z * z) / denom;
  } else {
    mAmp = 0.;
  }

  if ((TMath::Abs(q) < mQAccuracy) && (mAmp < 1.2 * maxSample)) { // converged and estimated amplitude is not mush larger than Max
    mTime = z / mDecTime;
    mChi2 = (y2 - 0.25 * exp(2. + z) * mAmp * (b2 - 2 * b1 * z + b0 * z2)) / nSamples;
    return kOK;
  } else { // too big difference, fit failed
    mAmp = maxSample;
    mTime = 0; // First count in sample
    mChi2 = 999.;
    return kFitFailed;
  }
}
