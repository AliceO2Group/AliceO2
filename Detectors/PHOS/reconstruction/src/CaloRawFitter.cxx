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

#include <gsl/span>

#include "PHOSReconstruction/CaloRawFitter.h"
#include "PHOSBase/PHOSSimParams.h"

using namespace o2::phos;

CaloRawFitter::CaloRawFitter()
{

  mSpikeThreshold = o2::phos::PHOSSimParams::Instance().mSpikeThreshold;
  mBaseLine = o2::phos::PHOSSimParams::Instance().mBaseLine;
  mPreSamples = o2::phos::PHOSSimParams::Instance().mPreSamples;
}

CaloRawFitter::FitStatus CaloRawFitter::evaluate(gsl::span<short unsigned int> signal)
{

  //Pedestal analysis mode
  if (mPedestalRun) {
    int nPed = signal.size();
    float mean = 0.;
    float rms = 0.;
    for (auto a : signal) {
      mean += a;
      rms += a * a;
    }
    if (nPed > 0) {
      mean /= nPed;
      rms = rms / nPed - mean * mean;
      if (rms > 0.) {
        rms = sqrt(rms);
      }
    }
    mAmp = mean;
    mTime = rms; // only in Pedestal mode!
    mOverflow = false;
    mStatus = kOK;
    return kOK;
  }

  // Extract amplitude and time using maximum and k-level methods
  return evalKLevel(signal);
}

CaloRawFitter::FitStatus CaloRawFitter::evalKLevel(gsl::span<short unsigned int> signal) //const ushort *signal, int sigStart, int sigLength)
{
  // Calculate signal parameters (energy, time, quality) from array of samples
  // Energy is a maximum sample minus pedestal 9
  // Time is the first time bin
  // Signal overflows is there are at least 3 samples of the same amplitude above 900

  int sigLength = signal.size();
  if (sigLength == 0) {
    return kEmptyBunch;
  }

  float pedMean = 0;
  int nPed = 0;
  mMaxSample = 0;
  int nMax = 0; //number of consequitive maximal samples
  bool spike = false;
  mOverflow = false;

  int ap = -1, app = -1; //remember previous values to evaluate spikes
  for (auto it = signal.rbegin(); it != signal.rend(); ++it) {
    uint16_t a = *it;
    if (mPedSubtract) {
      if (nPed < mPreSamples) { //inverse signal time order
        nPed++;
        pedMean += a;
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
      spike |= (2 * ap - (a + app) > 2 * mSpikeThreshold);
    }
    app = ap;
    ap = a;
  }
  mAmp = (float)mMaxSample;

  if (spike) {
    mTime = -2;
    mOverflow = false;
    return kSpike;
  }

  if (mMaxSample > 900 && nMax > 2) {
    mOverflow = true;
  }

  float pedestal = 0;
  if (mPedSubtract) {
    if (nPed > 0) {
      pedMean /= nPed;
    } else {
      mAmp = 0.;
      mTime = -2;
      mOverflow = false;
      return kBadPedestal;
    }
  } else {
    pedMean = 0.;
  }

  mAmp -= pedMean;
  if (mAmp < mBaseLine) {
    mAmp = 0;
  }

  //Evaluate time
  mTime = -2;
  const int nLine = 6;       //Parameters of fitting
  const float eMinTOF = 10.; //Choosed from beam-test and cosmic analyis
  const float kAmp = 0.35;   //Result slightly depends on them, so no getters
  // Avoid too low peak:
  if (mAmp < eMinTOF) {
    return kOK; //use estimated time
  }

  // Find index posK (kLevel is a level of "timestamp" point Tk):
  int posK = sigLength - 1; //last point before crossing k-level
  float levelK = pedestal + kAmp * mAmp;
  while (signal[posK] <= levelK && posK >= 0) {
    posK--;
  }
  posK++;

  if (posK == 0 || posK == sigLength - 1) {
    return kNoTime; //
  }

  // Find crossing point by solving linear equation (least squares method)
  int np = 0;
  int iup = posK - 1;
  int idn = posK;
  double sx = 0., sy = 0., sxx = 0., sxy = 0.;
  double x, y;

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

  double det = np * sxx - sx * sx;
  if (det == 0) {
    return kNoTime;
  }
  if (np == 0) {
    return kEmptyBunch;
  }
  double c1 = (np * sxy - sx * sy) / det; //slope
  double c0 = (sy - c1 * sx) / np;        //offset
  if (c1 == 0) {
    return kNoTime;
  }

  // Find where the line cross kLevel:
  mTime += (levelK - c0) / c1 - 5.; //5: mean offset between k-Level and start times

  if (mOverflow) {
    return kOverflow;
  } else {
    return kOK;
  }
}
