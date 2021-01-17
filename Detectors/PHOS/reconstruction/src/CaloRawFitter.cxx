// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CaloRawFitter.cxx
/// \author Dmitri Peresunko

#include "FairLogger.h"
#include <gsl/span>

#include "PHOSReconstruction/Bunch.h"
#include "PHOSReconstruction/CaloRawFitter.h"
#include "PHOSBase/PHOSSimParams.h"

using namespace o2::phos;

CaloRawFitter::FitStatus CaloRawFitter::evaluate(const std::vector<Bunch>& bunchlist)
{

  mAmp.clear();  // used as mean in this mode
  mTime.clear(); // used as RMS in pedestal mode
  mChi2.clear(); // used as mean in this mode

  //Pedestal analysis mode
  if (mPedestalRun) {
    int nPed = 0;
    float mean = 0.;
    float rms = 0.;
    for (auto b : bunchlist) {
      const std::vector<uint16_t>& signal = b.getADC();
      for (std::vector<uint16_t>::const_reverse_iterator it = signal.rbegin(); it != signal.rend(); ++it) {
        uint16_t a = *it;
        mean += a;
        rms += a * a;
        ++nPed;
      }
      if (nPed > 0) {
        mean /= nPed;
        rms = rms / nPed - mean * mean;
        if (rms > 0.) {
          rms = sqrt(rms);
        }
      }
    }
    mAmp.push_back(mean);
    mTime.push_back(rms);
    mStatus = kOK;
    return kOK;
  }

  for (auto b : bunchlist) {
    // Extract amplitude and time using maximum and k-level methods
    FitStatus s = evalKLevel(b);
    mStatus = s;

    // If this is Low Gain channel or explicitely requested, fit sample with Gamma2 function
    if (makeFit || (mLowGain && s == kOverflow)) {
      FitStatus s2 = fitGamma2(b);
    }
    //TODO: should we have separate status: overflow & fit OK, and overflow & fit failed, etc?
  }
  return mStatus;
}

CaloRawFitter::FitStatus CaloRawFitter::evalKLevel(const Bunch& b) //const ushort *signal, int sigStart, int sigLength)
{
  // Calculate signal parameters (energy, time, quality) from array of samples
  // Energy is a maximum sample minus pedestal 9
  // Time is the first time bin
  // Signal overflows is there are at least 3 samples of the same amplitude above 900

  float amp = 0.;
  float time = 0.;
  bool overflow = false;

  int sigLength = b.getBunchLength();
  if (sigLength == 0) {
    return kEmptyBunch;
  }

  const short kSpikeThreshold = o2::phos::PHOSSimParams::Instance().mSpikeThreshold;
  const short kBaseLine = o2::phos::PHOSSimParams::Instance().mBaseLine;
  const short kPreSamples = o2::phos::PHOSSimParams::Instance().mPreSamples;

  float pedMean = 0;
  float pedRMS = 0;
  int nPed = 0;
  mMaxSample = 0;
  int nMax = 0; //number of consequitive maximal samples
  bool spike = false;

  const std::vector<uint16_t>& signal = b.getADC();

  int ap = -1, app = -1; //remember previous values to evaluate spikes

  for (std::vector<uint16_t>::const_reverse_iterator it = signal.rbegin(); it != signal.rend(); ++it) {
    uint16_t a = *it;
    if (mPedSubtract) {
      if (nPed < kPreSamples) { //inverse signal time order
        nPed++;
        pedMean += a;
        pedRMS += a * a;
      }
    }
    if (a > mMaxSample) {
      mMaxSample = a;
      nMax = 0;
    }
    if (a == mMaxSample) {
      nMax++;
    }
    //check if there is a spike
    if (app >= 0 && ap >= 0) {
      spike |= (2 * ap - (a + app) > 2 * kSpikeThreshold);
    }
    app = ap;
    ap = a;
  }
  amp = (float)mMaxSample;

  if (spike) {
    mAmp.push_back(amp);
    mTime.push_back(b.getStartTime() - 2);
    mOverflow.push_back(false);
    return kSpike;
  }

  if (mMaxSample > 900 && nMax > 2) {
    overflow = true;
  }

  float pedestal = 0;
  if (mPedSubtract) {
    if (nPed > 0) {
      pedRMS = (pedRMS - pedMean * pedMean / nPed) / nPed;
      if (pedRMS > 0.) {
        pedRMS = sqrt(pedRMS);
      }
      pedestal = pedMean / nPed;
    } else {
      mAmp.push_back(0);
      mTime.push_back(b.getStartTime() - 2);
      mOverflow.push_back(false);
      return kBadPedestal;
    }
  }

  amp -= pedestal;
  if (amp < kBaseLine) {
    amp = 0;
  }

  //Evaluate time
  time = b.getStartTime() - 2;
  const int nLine = 6;       //Parameters of fitting
  const float eMinTOF = 10.; //Choosed from beam-test and cosmic analyis
  const float kAmp = 0.35;   //Result slightly depends on them, so no getters
  // Avoid too low peak:
  if (amp < eMinTOF) {
    mAmp.push_back(amp);
    mTime.push_back(time);
    mOverflow.push_back(false);
    return kOK; //use estimated time
  }

  // Find index posK (kLevel is a level of "timestamp" point Tk):
  int posK = sigLength - 1; //last point before crossing k-level
  float levelK = pedestal + kAmp * amp;
  while (signal[posK] <= levelK && posK >= 0) {
    posK--;
  }
  posK++;

  if (posK == 0 || posK == sigLength - 1) {
    mAmp.push_back(amp);
    mTime.push_back(time);
    mOverflow.push_back(false);
    return kNoTime; //
  }

  // Find crossing point by solving linear equation (least squares method)
  int np = 0;
  int iup = posK - 1;
  int idn = posK;
  Double_t sx = 0., sy = 0., sxx = 0., sxy = 0.;
  Double_t x, y;

  while (np < nLine) {
    //point above crossing point
    if (iup >= 0) {
      x = sigLength - iup - 1;
      y = signal[iup];
      sx += x;
      sy += y;
      sxx += (x * x);
      sxy += (x * y);
      np++;
      iup--;
    }
    //Point below crossing point
    if (idn < sigLength) {
      if (signal[idn] < pedestal) {
        idn = sigLength - 1; //do not scan further
        idn++;
        continue;
      }
      x = sigLength - idn - 1;
      y = signal[idn];
      sx += x;
      sy += y;
      sxx += (x * x);
      sxy += (x * y);
      np++;
      idn++;
    }
    if (idn >= sigLength && iup < 0) {
      break; //can not fit futher
    }
  }

  Double_t det = np * sxx - sx * sx;
  if (det == 0) {
    mAmp.push_back(amp);
    mTime.push_back(time);
    mOverflow.push_back(false);
    return kNoTime;
  }
  if (np == 0) {
    mAmp.push_back(amp);
    mTime.push_back(time);
    mOverflow.push_back(false);
    return kEmptyBunch;
  }
  Double_t c1 = (np * sxy - sx * sy) / det; //slope
  Double_t c0 = (sy - c1 * sx) / np;        //offset
  if (c1 == 0) {
    mAmp.push_back(amp);
    mTime.push_back(time);
    mOverflow.push_back(false);
    return kNoTime;
  }

  // Find where the line cross kLevel:
  time += (levelK - c0) / c1 - 5.; //5: mean offset between k-Level and start times

  mAmp.push_back(amp);
  mTime.push_back(time);
  if (overflow) {
    mOverflow.push_back(true);
    return kOverflow;
  } else {
    mOverflow.push_back(false);
    return kOK;
  }
}

CaloRawFitter::FitStatus CaloRawFitter::fitGamma2(const Bunch& b)
{
  // Fit bunch with gamma2 function
  // TODO!!! validate method
  //initial values
  float A = mAmp.back();
  float t0 = mTime.back();
  const std::vector<uint16_t>& signal = b.getADC();
  uint16_t tsart = b.getStartTime();

  const float alpha = 17.;      //Decay time in units of 100ns  //TODO!!! to be adjusted
  const float kEpsilon = 1.e-6; //Accuracy of fit //TODO!!! to be adjusted
  const int nIter = 10;         //Maximal number of iterations //TODO!!! to be adjusted

  float chi2 = 0;
  float derT = 0, derTprev = 0.;
  float stepT = 0.1;
  int iter = 0, i = 0;
  do {
    chi2 = 0.;
    derTprev = derT;
    derT = 0.;
    i = 0;
    float sA = 0., sB = 0.;
    std::vector<uint16_t>::const_reverse_iterator it = signal.rbegin();
    while (it != signal.rend()) {
      uint16_t si = *it;
      float ti = tsart + i - t0;
      it++;
      i++;
      if (mOverflow.back() && si == mMaxSample) { //do not fit saturated samples
        continue;
      }
      float fi = ti * ti * exp(-ti * alpha);
      chi2 += (si - A * fi) * (si - A * fi);
      sA += si * fi;
      sB += fi * fi;
      if (ti != 0.) {
        derT += (2. / ti - alpha) * fi * (A * fi - si);
      }
    }
    derT *= A;
    //calculate time step and next time
    if (derTprev != 0. && derT - derTprev != 0.) {
      stepT = derT / (derT - derTprev) * stepT;
    }
    derTprev = derT;
    t0 -= stepT;
    if (sB > 0.) {
      A = sA / sB;
    }
  } while (fabs(stepT) > kEpsilon && iter < nIter); // if time step is too small, stop

  if (iter >= nIter) { //Fit did not converge, keep old A and t0.
    return kFitFailed;
  }

  if (i > 0) { //chi2/NDF
    chi2 /= i;
  }

  mTime.back() = t0;
  mAmp.back() = A;
  return kOK;
}
