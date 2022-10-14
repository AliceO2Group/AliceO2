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

/// \file CaloRawFitterStandard.cxx
/// \author Hadi Hassan (hadi.hassan@cern.ch)

#include <fairlogger/Logger.h>
#include <random>

// ROOT sytem
#include "TMath.h"
#include "TF1.h"
#include "TGraph.h"

#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/CaloFitResults.h"
#include "DataFormatsEMCAL/Constants.h"

#include "EMCALReconstruction/CaloRawFitterStandard.h"

using namespace o2::emcal;

CaloRawFitterStandard::CaloRawFitterStandard() : CaloRawFitter("Chi Square ( Standard )", "Standard")
{
  mAlgo = FitAlgorithm::Standard;
}

double CaloRawFitterStandard::rawResponseFunction(double* x, double* par)
{
  double signal = 0.;
  double tau = par[2];
  double n = par[3];
  double ped = par[4];
  double xx = (x[0] - par[1] + tau) / tau;

  if (xx <= 0) {
    signal = ped;
  } else {
    signal = ped + par[0] * TMath::Power(xx, n) * TMath::Exp(n * (1 - xx));
  }

  return signal;
}

CaloFitResults CaloRawFitterStandard::evaluate(const gsl::span<const Bunch> bunchlist)
{
  float time = 0;
  float amp = 0;
  float chi2 = 0;
  int ndf = 0;
  bool fitDone = kFALSE;

  auto [nsamples, bunchIndex, ampEstimate,
        maxADC, timeEstimate, pedEstimate, first, last] = preFitEvaluateSamples(bunchlist, mAmpCut);

  if (bunchIndex >= 0 && ampEstimate >= mAmpCut) {
    time = timeEstimate;
    int timebinOffset = bunchlist[bunchIndex].getStartTime() - (bunchlist[bunchIndex].getBunchLength() - 1);
    amp = ampEstimate;

    if (nsamples > 1 && maxADC < constants::OVERFLOWCUT) {
      try {
        std::tie(amp, time, chi2) = fitRaw(first, last);
        time += timebinOffset;
        timeEstimate += timebinOffset;
        ndf = nsamples - 2;
        fitDone = true;
      } catch (RawFitterError_t& error) {
      }
    }
  }
  if (fitDone) {
    float ampAsymm = (amp - ampEstimate) / (amp + ampEstimate);
    float timeDiff = time - timeEstimate;

    if ((TMath::Abs(ampAsymm) > 0.1) || (TMath::Abs(timeDiff) > 2)) {
      amp = ampEstimate;
      time = timeEstimate;
      fitDone = kFALSE;
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
  throw RawFitterError_t::FIT_ERROR;
}

std::tuple<float, float, float> CaloRawFitterStandard::fitRaw(int firstTimeBin, int lastTimeBin) const
{

  float amp(0), time(0), chi2(0);

  int nsamples = lastTimeBin - firstTimeBin + 1;
  if (nsamples < 3) {
    throw RawFitterError_t::FIT_ERROR;
  }

  TGraph gSig(nsamples);

  for (int i = 0; i < nsamples; i++) {
    int timebin = firstTimeBin + i;
    gSig.SetPoint(i, timebin, getReversed(timebin));
  }

  TF1 signalF("signal", CaloRawFitterStandard::rawResponseFunction, 0, constants::EMCAL_MAXTIMEBINS, 5);

  signalF.SetParameters(10., 5., constants::TAU, constants::ORDER, 0.); // set all defaults once, just to be safe
  signalF.SetParNames("amp", "t0", "tau", "N", "ped");
  signalF.FixParameter(2, constants::TAU);
  signalF.FixParameter(3, constants::ORDER);
  signalF.FixParameter(4, 0);
  signalF.SetParameter(1, time);
  signalF.SetParameter(0, amp);
  signalF.SetParLimits(0, 0.5 * amp, 2 * amp);
  signalF.SetParLimits(1, time - 4, time + 4);

  int status = gSig.Fit(&signalF, "QROW"); // Note option 'W': equal errors on all points
  if (status == 0) {
    amp = signalF.GetParameter(0);
    time = signalF.GetParameter(1);
    chi2 = signalF.GetChisquare();
  } else {
    throw RawFitterError_t::FIT_ERROR;
  }

  return std::make_tuple(amp, time, chi2);
}
