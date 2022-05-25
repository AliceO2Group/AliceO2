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

/// \file CaloRawFitterGamma2.cxx
/// \author Martin Poghosyan (Martin.Poghosyan@cern.ch)

#include "FairLogger.h"
#include <cfloat>
#include <random>

// ROOT sytem
#include "TMath.h"

#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/CaloFitResults.h"
#include "DataFormatsEMCAL/Constants.h"

#include "EMCALReconstruction/CaloRawFitterGamma2.h"

using namespace o2::emcal;

CaloRawFitterGamma2::CaloRawFitterGamma2() : CaloRawFitter("Chi Square ( Gamma2 )", "Gamma2")
{
  mAlgo = FitAlgorithm::Gamma2;
}

CaloFitResults CaloRawFitterGamma2::evaluate(const gsl::span<const Bunch> bunchlist)
{
  float time = 0;
  float amp = 0;
  float chi2 = 0;
  int ndf = 0;
  bool fitDone = false;

  auto [nsamples, bunchIndex, ampEstimate,
        maxADC, timeEstimate, pedEstimate, first, last] = preFitEvaluateSamples(bunchlist, mAmpCut);

  if (bunchIndex >= 0 && ampEstimate >= mAmpCut) {
    time = timeEstimate;
    int timebinOffset = bunchlist[bunchIndex].getStartTime() - (bunchlist[bunchIndex].getBunchLength() - 1);
    amp = ampEstimate;

    if (nsamples > 2 && maxADC < constants::OVERFLOWCUT) {
      std::tie(amp, time) = doParabolaFit(timeEstimate - 1);
      mNiter = 0;
      try {
        chi2 = doFit_1peak(first, nsamples, amp, time);
        fitDone = true;
      } catch (RawFitterError_t& e) {
        // Fit has failed, set values to estimates
        // TODO: Check whether we want to include cases in which the peak fit failed
        amp = ampEstimate;
        time = timeEstimate;
        chi2 = 1.e9;
      }

      time += timebinOffset;
      timeEstimate += timebinOffset;
      ndf = nsamples - 2;
    }
  }

  if (fitDone) {
    float ampAsymm = (amp - ampEstimate) / (amp + ampEstimate);
    float timeDiff = time - timeEstimate;

    if ((TMath::Abs(ampAsymm) > 0.1) || (TMath::Abs(timeDiff) > 2)) {
      amp = ampEstimate;
      time = timeEstimate;
      fitDone = false;
    }
  }
  if (amp >= mAmpCut) {
    if (!fitDone) {
      std::default_random_engine generator;
      std::uniform_real_distribution<float> distribution(0.0, 1.0);
      amp += (0.5 - distribution(generator));
    }
    time = time * constants::EMCAL_TIMESAMPLE;
    time -= mL1Phase;

    return CaloFitResults(maxADC, pedEstimate, 0, amp, time, (int)time, chi2, ndf);
  }
  // Fit failed, rethrow error
  throw RawFitterError_t::FIT_ERROR;
}

float CaloRawFitterGamma2::doFit_1peak(int firstTimeBin, int nSamples, float& ampl, float& time)
{

  float chi2(0.);

  // fit using gamma-2 function   (ORDER =2 assumed)
  if (nSamples < 3) {
    throw RawFitterError_t::FIT_ERROR;
  }
  if (mNiter > mNiterationsMax) {
    throw RawFitterError_t::FIT_ERROR;
  }

  double D, dA, dt;
  double c11 = 0;
  double c12 = 0;
  double c21 = 0;
  double c22 = 0;
  double d1 = 0;
  double d2 = 0;

  mNiter++;

  for (int itbin = 0; itbin < nSamples; itbin++) {

    double ti = (itbin - time) / constants::TAU;
    if ((ti + 1) < 0) {
      continue;
    }

    double g_1i = (ti + 1) * TMath::Exp(-2 * ti);
    double g_i = (ti + 1) * g_1i;
    double gp_i = 2 * (g_i - g_1i);
    double q1_i = (2 * ti + 1) * TMath::Exp(-2 * ti);
    double q2_i = g_1i * g_1i * (4 * ti + 1);
    c11 += (getReversed(itbin) - ampl * 2 * g_i) * gp_i;
    c12 += g_i * g_i;
    c21 += getReversed(itbin) * q1_i - ampl * q2_i;
    c22 += g_i * g_1i;
    double delta = ampl * g_i - getReversed(itbin);
    d1 += delta * g_i;
    d2 += delta * g_1i;
    chi2 += (delta * delta);
  }

  D = c11 * c22 - c12 * c21;

  if (TMath::Abs(D) < DBL_EPSILON) {
    throw RawFitterError_t::FIT_ERROR;
  }

  dt = (d1 * c22 - d2 * c12) / D * constants::TAU;
  dA = (d1 * c21 - d2 * c11) / D;

  time += dt;
  ampl += dA;

  if (TMath::Abs(dA) > 1 || TMath::Abs(dt) > 0.01) {
    chi2 = doFit_1peak(firstTimeBin, nSamples, ampl, time);
  }

  return chi2;
}

std::tuple<float, float> CaloRawFitterGamma2::doParabolaFit(int maxTimeBin) const
{
  float amp(0.), time(0.);

  // The equation of parabola is "y = a*x^2 + b*x + c"
  // We have to find "a", "b", and "c"

  double a = (getReversed(maxTimeBin + 2) + getReversed(maxTimeBin) - 2. * getReversed(maxTimeBin + 1)) / 2.;

  if (TMath::Abs(a) < DBL_EPSILON) {
    amp = getReversed(maxTimeBin + 1);
    time = maxTimeBin + 1;
    return std::make_tuple(amp, time);
  }

  double b = getReversed(maxTimeBin + 1) - getReversed(maxTimeBin) - a * (2. * maxTimeBin + 1);
  double c = getReversed(maxTimeBin) - b * maxTimeBin - a * maxTimeBin * maxTimeBin;

  time = -b / 2. / a;
  amp = a * time * time + b * time + c;

  return std::make_tuple(amp, time);
}
