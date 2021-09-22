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

/// \file CaloRawFitter.cxx
/// \author Hadi Hassan (hadi.hassan@cern.ch)
#include <numeric>
#include <gsl/span>

// ROOT sytem
#include "TMath.h"

#include "FairLogger.h"
#include "DataFormatsEMCAL/Constants.h"
#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/CaloFitResults.h"
#include "EMCALReconstruction/CaloRawFitter.h"

using namespace o2::emcal;

//Default constructor
CaloRawFitter::CaloRawFitter(const char* name, const char* nameshort) : mMinTimeIndex(-1),
                                                                        mMaxTimeIndex(-1),
                                                                        mAmpCut(4),
                                                                        mNsamplePed(3),
                                                                        mIsZerosupressed(false),
                                                                        mName(name),
                                                                        mNameShort(nameshort),
                                                                        mAlgo(FitAlgorithm::NONE),
                                                                        mL1Phase(0),
                                                                        mAmp(0)
{
}

void CaloRawFitter::setTimeConstraint(int min, int max)
{

  if ((min > max) || min > constants::EMCAL_MAXTIMEBINS || max > constants::EMCAL_MAXTIMEBINS) {
    LOG(WARN) << "Attempt to set Invalid time bin range (Min , Max) = (" << min << ", " << max << "), Ingored";
  } else {
    mMinTimeIndex = min;
    mMaxTimeIndex = max;
  }
}

unsigned short CaloRawFitter::getMaxAmplitudeBunch(const gsl::span<unsigned short> data) const
{
  return *std::max_element(data.begin(), data.end());
}

std::tuple<int, int> CaloRawFitter::findPeakRegion(const gsl::span<double> adcValues, short indexMaxADC, int threshold) const
{

  int first(0), last(0);
  int tmpfirst = indexMaxADC;
  int tmplast = indexMaxADC;
  double prevFirst = adcValues[indexMaxADC];
  double prevLast = adcValues[indexMaxADC];
  bool firstJump = false;
  bool lastJump = false;

  while ((tmpfirst >= 0) && (adcValues[tmpfirst] >= threshold) && (!firstJump)) {
    // jump check:
    if (tmpfirst != indexMaxADC) { // neighbor to maxindex can share peak with maxindex
      if (adcValues[tmpfirst] >= prevFirst) {
        firstJump = true;
      }
    }
    prevFirst = adcValues[tmpfirst];
    tmpfirst--;
  }

  while ((tmplast < adcValues.size()) && (adcValues[tmplast] >= threshold) && (!lastJump)) {
    // jump check:
    if (tmplast != indexMaxADC) { // neighbor to maxindex can share peak with maxindex
      if (adcValues[tmplast] >= prevLast) {
        lastJump = true;
      }
    }
    prevLast = adcValues[tmplast];
    tmplast++;
  }

  // we keep one pre- or post- sample if we can (as in online)
  // check though if we ended up on a 'jump', or out of bounds: if so, back up
  if (firstJump || tmpfirst < 0) {
    tmpfirst++;
  }
  if (lastJump || tmplast >= adcValues.size()) {
    tmplast--;
  }

  first = tmpfirst;
  last = tmplast;

  return std::make_tuple(first, last);
}

std::optional<std::tuple<float, std::array<double, constants::EMCAL_MAXTIMEBINS>>> CaloRawFitter::reverseAndSubtractPed(const Bunch& bunch) const
{
  std::array<double, constants::EMCAL_MAXTIMEBINS> outarray;
  int length = bunch.getBunchLength();
  const gsl::span<const uint16_t> sig(bunch.getADC());

  double ped = 0.;
  if (mIsZerosupressed) {
    auto pedestalResult = evaluatePedestal(sig, length);
    if (pedestalResult.has_value()) {
      ped = pedestalResult.value();
    } else {
      std::optional<std::tuple<float, std::array<double, constants::EMCAL_MAXTIMEBINS>>> empty;
      return empty;
    }
  }

  for (int i = 0; i < length; i++) {
    outarray[i] = sig[length - i - 1] - ped;
  }

  return std::make_tuple(ped, outarray);
}

std::optional<double> CaloRawFitter::evaluatePedestal(const gsl::span<const uint16_t> data, std::optional<int> length) const
{
  if (!mNsamplePed) {
    return std::optional<double>();
  }
  return static_cast<double>(std::accumulate(data.begin(), data.begin() + mNsamplePed, 0)) / mNsamplePed;
}

std::tuple<short, int> CaloRawFitter::getMaxAmplitudeBunch(const Bunch& bunch) const
{
  short maxADC = -1;
  int maxIndex = -1;
  const std::vector<uint16_t>& sig = bunch.getADC();

  for (int i = 0; i < bunch.getBunchLength(); i++) {
    if (sig[i] > maxADC) {
      maxADC = sig[i];
      maxIndex = i;
    }
  }

  return std::make_tuple(maxADC, bunch.getBunchLength() - 1 - maxIndex + bunch.getStartTime());
}

bool CaloRawFitter::isMaxADCBunchEdge(const Bunch& bunch) const
{
  short maxADC = -1;
  int indexMax = -1;
  const std::vector<uint16_t>& sig = bunch.getADC();

  for (int i = 0; i < bunch.getBunchLength(); i++) {
    if (sig[i] > maxADC) {
      maxADC = sig[i];
      indexMax = i;
    }
  }

  bool isBunchEdge = false;
  if (indexMax == 0 || indexMax == (bunch.getBunchLength() - 1)) {
    isBunchEdge = true;
  }

  return isBunchEdge;
}

std::optional<std::tuple<short, short, short>> CaloRawFitter::selectMaximumBunch(const gsl::span<const Bunch>& bunchvector)
{
  short bunchindex = -1;
  short indexMaxInBunch(0), maxADCallBunches(-1);

  for (unsigned int i = 0; i < bunchvector.size(); i++) {
    auto [maxADC, maxIndex] = getMaxAmplitudeBunch(bunchvector[i]); // CRAP PTH, bug fix, trouble if more than one bunches
    if (isInTimeRange(maxIndex, mMaxTimeIndex, mMinTimeIndex)) {
      if (maxADC > maxADCallBunches) {
        bunchindex = i;
        indexMaxInBunch = maxIndex;
        maxADCallBunches = maxADC;
      }
    }
  }
  if (bunchindex >= 0) {
    // reject bunch if the max. ADC value is at the edges of a bunch
    if (isMaxADCBunchEdge(bunchvector[bunchindex])) {
      std::optional<std::tuple<short, short, short>> emptyresult;
      return emptyresult;
    }
  }

  return std::make_tuple(bunchindex, indexMaxInBunch, maxADCallBunches);
}

bool CaloRawFitter::isInTimeRange(int indexMaxADC, int maxtime, int mintime) const
{
  if ((mintime < 0 && maxtime < 0) || maxtime < 0) {
    return true;
  }

  return (indexMaxADC < maxtime) && (indexMaxADC > mintime) ? true : false;
}

std::optional<double> CaloRawFitter::calculateChi2(double amp, double time,
                                                   int first, int last,
                                                   double adcErr, double tau) const
{
  if (first == last || first < 0) { // signal consists of single sample, chi2 estimate (0) not too well defined..
                                    // or, first is negative, the indices are not valid
    return std::optional<double>();
  }

  int nsamples = last - first + 1;

  double chi2 = 0;

  for (int i = 0; i < nsamples; i++) {
    int x = first + i;                  // timebin
    double xx = (x - time + tau) / tau; // help variable
    double f = 0.0;

    if (xx > 0) {
      f = amp * xx * xx * TMath::Exp(2 * (1 - xx));
    }

    double dy = mReversed[x] - f;
    chi2 += dy * dy;
  }

  if (adcErr > 0.0) { // weight chi2
    chi2 /= (adcErr * adcErr);
  }

  return chi2;
}

CaloRawFitter::PreFitResults CaloRawFitter::preFitEvaluateSamples(const gsl::span<const Bunch> bunchvector, int adcThreshold)
{

  int nsamples(0), first(0), last(0), indexMaxADCRReveresed(0);
  double peakADC(0.), pedestal(0.);

  PreFitResults infos;

  // Reset buffer for reversed bunch, no matter whether the bunch could be selected or not
  mReversed.fill(0);

  // select the bunch with the highest amplitude unless any time constraints is set
  auto maxBunchSelection = selectMaximumBunch(bunchvector);
  if (!maxBunchSelection.has_value()) {
    infos.mErrorCode = PreFitError_t::BUNCH_NOT_SELECTED;
    return infos;
  }
  auto [bunchindex, indexMaxADC, adcMAX] = maxBunchSelection.value();

  // something valid was found, and non-zero amplitude
  if (bunchindex >= 0) {
    infos.mIndexMaxBunch = bunchindex;
    infos.mADC = adcMAX;
    if (adcMAX >= adcThreshold) {
      // use more convenient numbering and possibly subtract pedestal

      //std::tie(ped, mReversed) = reverseAndSubtractPed((bunchvector.at(index)), altrocfg1, altrocfg2);
      //maxf = (float)*std::max_element(mReversed.begin(), mReversed.end());

      int bunchlength = bunchvector[bunchindex].getBunchLength();
      const std::vector<uint16_t>& sig = bunchvector[bunchindex].getADC();

      if (!mIsZerosupressed) {
        auto pedestalTmp = evaluatePedestal(sig, bunchlength);
        if (!pedestalTmp.has_value()) {
          infos.mErrorCode = PreFitError_t::PEDESTAL_ERROR;
          return infos;
        }
        pedestal = pedestalTmp.value();
      }

      int testindexReverse = -1;
      for (int i = 0; i < bunchlength; i++) {
        mReversed[i] = sig[bunchlength - i - 1] - pedestal;
        if (mReversed[i] > peakADC) {
          peakADC = mReversed[i];
          testindexReverse = i;
        }
      }
      infos.mPedestal = pedestal;
      infos.mMaxAmplitude = peakADC;

      if (peakADC >= adcThreshold) // possibly significant signal
      {
        // select array around max to possibly be used in fit
        indexMaxADCRReveresed = indexMaxADC - bunchvector[bunchindex].getStartTime();
        infos.mIndexMaxAmplitudeArray = indexMaxADCRReveresed;
        std::tie(first, last) = findPeakRegion(gsl::span<double>(&mReversed[0], bunchvector[bunchindex].getBunchLength()), indexMaxADCRReveresed, adcThreshold);
        infos.mFirstTimebin = first;
        infos.mLastTimebin = last;

        // sanity check: maximum should not be in first or last bin
        // if we should do a fit
        if (first != indexMaxADCRReveresed && last != indexMaxADCRReveresed) {
          // calculate how many samples we have
          infos.mSampleLength = last - first + 1;
        }
      }
    } else {
      infos.mErrorCode = PreFitError_t::BUNCH_LOW_SIGNAL;
      return infos;
    }
  } else {
    infos.mErrorCode = PreFitError_t::BUNCH_NOT_SELECTED;
    return infos;
  }

  return infos;
}

CaloFitResults CaloRawFitter::buildErrorResultsForPrefit(CaloRawFitter::PreFitError_t error)
{
  CaloFitResults::RawFitterError_t fiterror = CaloFitResults::RawFitterError_t::NO_ERROR;
  switch (error) {
    case PreFitError_t::BUNCH_LOW_SIGNAL:
      fiterror = CaloFitResults::RawFitterError_t::LOW_SIGNAL;
      break;
    case PreFitError_t::BUNCH_NOT_SELECTED:
      fiterror = CaloFitResults::RawFitterError_t::BUNCH_NOT_OK;
      break;
    case PreFitError_t::PEDESTAL_ERROR:
      fiterror = CaloFitResults::RawFitterError_t::SAMPLE_UNINITIALIZED;
      break;
  }
  return CaloFitResults(fiterror);
}
