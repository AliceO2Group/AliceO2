// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  mTimeAccuracy = o2::phos::PHOSSimParams::Instance().mSampleTimeFitAccuracy;
  mAmpAccuracy = o2::phos::PHOSSimParams::Instance().mSampleAmpFitAccuracy;
  init();
}

void CaloRawFitterGS::init()
{
  //prepare fitting arrays, once per lifetime
  double k = o2::phos::PHOSSimParams::Instance().mSampleDecayTime;
  ma0[0] = 1.;
  mb0[0] = 1.;
  mb1[0] = 0.;
  mb2[0] = 0.;
  mb3[0] = 0.;
  mb4[0] = 0.;
  for (int i = 1; i < NMAXSAMPLES; i++) {
    double xi = k * i;
    ma0[i] = exp(-xi);
    mb0[i] = mb0[i - 1] + ma0[i];
    mb1[i] = 4 * mb1[i - 1] + ma0[i] * xi;
    mb2[i] = 6 * mb2[i - 1] + ma0[i] * xi * xi;
    mb3[i] = 4 * mb3[i - 1] + ma0[i] * xi * xi * xi;
    mb4[i] = mb4[i - 1] + ma0[i] * xi * xi * xi * xi;
  }
}

CaloRawFitterGS::FitStatus CaloRawFitterGS::evaluate(gsl::span<short unsigned int> signal)
{

  //Pedestal analysis mode
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
  // Energy is a maximum sample minus pedestal 9
  // Time is the first time bin
  // Signal overflows is there are at least 3 samples of the same amplitude above 900

  int sigLength = signal.size();
  if (sigLength == 0) {
    return kEmptyBunch;
  }
  mAmp = 0.;
  mTime = 0.;
  mChi2 = 0.;
  mOverflow = false;
  FitStatus status = kNotEvaluated;

  //if pedestal should be subtracted first evaluate it
  float pedMean = 0;
  int nPed = 0;
  if (mPedSubtract) {
    //remember inverse time order
    for (auto it = signal.rbegin(); (nPed < mPreSamples) && it != signal.rend(); ++it) {
      nPed++;
      pedMean += *it;
    }
    if (nPed > 0) {
      pedMean /= nPed;
    }
  }

  float maxSample = 0.;  //maximal sample value
  int nMax = 0;          //number of consequitive maximal samples
  bool spike = false;    //did we observe spike?
  int ap = -1, app = -1; //remember previous values to evaluate spikes
  double At = 0., Bt = 0., Ct = 0., y2 = 0.;

  int nSamples = signal.size();
  if (mPedSubtract) {
    nSamples = -mPreSamples;
  }
  int firstS = nSamples - 1;
  for (int i = 0; i < nSamples; i++) {
    float a = signal[firstS - i] - pedMean; //remember inverse order of samples
    float xi = i * mDecTime;
    if (a > maxSample) {
      mMaxSample = a;
      nMax = 1;
    }
    if (a == maxSample) {
      nMax++;
    }
    //check if there was a spike in previous step?
    if (app >= 0 && ap >= 0) {
      spike = (2 * ap - (a + app) > 2 * mSpikeThreshold);
    }
    if (spike) {
      status = kSpike;
      //Try to recover: subtract last point contribution and replace by average of "app" and "a"
      float atmp = 0.5 * (app + a);
      float xiprev = xi - mDecTime;
      float ss = (atmp - ap) * ma0[i - 1];
      At += ss; //spike can not appear at 0-th bin
      Bt += ss * xiprev;
      Ct += ss * xiprev * xiprev;
      y2 += (atmp * atmp - ap * ap);
    } else {
      app = ap;
      ap = a;
    }
    //Check if in saturation
    if (maxSample > 900 && nMax > 3) {
      mOverflow = true;
    }

    //to calculate time
    float st = a * ma0[i];
    At += st;
    Bt += st * xi;
    Ct += st * xi * xi;
    y2 += a * a;
    //to calculate amplitude
  } //Scanned full sample

  //calculate time, amp and chi2
  double polB = At - Bt;
  double polC = Ct - 2. * Bt;
  if (At == 0.) {
    if (polB == 0.) {
      mTime = 999.;
      status = kFitFailed;
    } else {
      mTime = -polC / (2. * polB);
    }
  } else {
    double d = polB * polB - At * polC;
    if (d >= 0) {
      mTime = (-polB - sqrt(d)) / (At * mDecTime);
    } else {
      mTime = 999.;
      status = kFitFailed;
    }
  }
  if (status == kFitFailed && !mOverflow) { //in case of overflow try to recover
    mAmp = 0.;
    mTime = 999.;
    mChi2 = 999.;
    return status;
  }

  if (!mOverflow) { //normal sample, calculate amp and chi2 and return
    double tt = mTime * mDecTime;
    double tt2 = tt * tt;
    double expT = exp(tt);
    double nom = (At * tt2 - 2. * Bt * tt + Ct) * expT; //  1./(k*k) cancel with denom
    int i = nSamples - 1;
    double denom = (mb4[i] - tt * (mb3[i] - tt * (mb2[i] - tt * (mb1[i] - tt * mb0[i])))) *
                   expT * expT / (mDecTime * mDecTime); // 1/(k*k) cancel with nom
    if (denom != 0) {
      mAmp = nom / denom;
      mChi2 = (y2 - (2. * (At * tt2 - 2. * tt * Bt + Ct) - mAmp * denom * expT) * mAmp * expT / (mDecTime * mDecTime)) / (i + 1);
    } else {
      mAmp = 0.;
      status = kFitFailed;
    }
    return status;
  } else { // overflow: try iterative procedure but for lowGain only
    if (!mLowGain) {
      mAmp = 0.;
      mTime = 999.;
      mChi2 = 999.;
      return kOverflow;
    }

    //Try to recalculate parameters replacing overflow/spike values by those expected from the sample shape
    short nIter = 0;
    double timeOld = mTime;
    double ampOld = mAmp;
    if (status == kFitFailed) { //could not calculate time, amp: set best guess
      timeOld = 0;
      ampOld = maxSample;
    }

    //Iterative process, not more than NITERATIONS
    short nMaxIter = o2::phos::PHOSSimParams::Instance().mNIterations;
    for (short nIter = 0; nIter < nMaxIter; nIter++) {
      ap = -1;
      app = -1; //remember previous values to evaluate spikes

      double expT = exp(mDecTime * timeOld);
      for (int i = 0; i < nSamples; i++) {
        float a = signal[firstS - i] - pedMean; //remember inverse order of samples
        float xi = i * mDecTime;
        if (a == maxSample) { //overflow, replace with calculated
          a = ampOld * ma0[i] * (timeOld - i) * (timeOld - i) * expT;
        }
        //check if there was a spike in prev step?
        if (app >= 0 && ap >= 0) {
          if (2 * ap - (a + app) > 2 * mSpikeThreshold) {
            //Try to recover: subtract last point contribution and replace by average of "app" and "a"
            float atmp = ampOld * ma0[i] * (timeOld - i + 1) * (timeOld - i + 1) * expT; //0.5*(app+a) ;
            float xiprev = xi - mDecTime;
            float s = (atmp - ap) * ma0[i - 1]; //spike can not appear at 0-th bin
            At += s;
            Bt += s * xiprev;
            Ct += s * xiprev * xiprev;
            y2 += (atmp * atmp - ap * ap);
          }
        } else {
          app = ap;
          ap = a;
        }

        //to calculate time
        float ss = a * ma0[i];
        At += ss;
        Bt += ss * xi;
        Ct += ss * xi * xi;
        y2 += a * a;
      }
      //evaluate new time and amp

      double polB = At - Bt;
      double polC = Ct - 2. * Bt;
      if (At == 0.) {
        if (polB == 0.) {
          mTime = 999.;
          status = kFitFailed;
        } else {
          mTime = -polC / (2. * polB);
        }
      } else {
        double d = polB * polB - At * polC;
        if (d >= 0) {
          mTime = (-polB - sqrt(d)) / (At * mDecTime);
        } else {
          mTime = 999.;
          status = kFitFailed;
        }
      }
      if (status == kFitFailed) { //Can not improve, give up
        mAmp = 0;
        mTime = 999.;
        mChi2 = 999.;
        mOverflow = false;
        return status;
      }
      double tt = mTime * mDecTime;
      expT = exp(tt);
      double nom = (At * tt * tt - 2. * Bt * tt + Ct) * expT; //  1./(k*k) cancel with denom
      int i = nSamples - 1;
      double denom = ((mb4[i] - tt * (mb3[i] - tt * (mb2[i] - tt * (mb1[i] - tt * mb0[i]))))) *
                     expT * expT / (mDecTime * mDecTime); // 1/(k*k) cancel with nom
      if (denom != 0) {
        mAmp = nom / denom;
        mChi2 = (y2 - (2. * (At * tt * tt - 2. * tt * Bt + Ct) - mAmp * denom * expT) * mAmp * expT / (mDecTime * mDecTime)) / (i + 1);
      } else {
        mAmp = 0;
        mTime = 999.;
        mChi2 = 999.;
        mOverflow = false;
        return kFitFailed;
      }

      //Check modification and quit if ready
      if (abs(mTime - timeOld) < mTimeAccuracy && abs(mAmp - ampOld) < ampOld * mAmpAccuracy) {
        break;
      }

      timeOld = mTime;
      ampOld = mAmp;
    }
  }

  return status;
}
