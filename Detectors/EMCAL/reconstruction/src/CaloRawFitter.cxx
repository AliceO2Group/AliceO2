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
#include <cassert>
#include <numeric>
#include <gsl/span>

// ROOT sytem
#include "TMath.h"

#include <fairlogger/Logger.h>
#include "DataFormatsEMCAL/Constants.h"
#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/CaloFitResults.h"
#include "EMCALReconstruction/CaloRawFitter.h"

using namespace o2::emcal;

int CaloRawFitter::getErrorNumber(CaloRawFitter::RawFitterError_t fiterror)
{
  switch (fiterror) {
    case RawFitterError_t::SAMPLE_UNINITIALIZED:
      return 0;
    case RawFitterError_t::FIT_ERROR:
      return 1;
    case RawFitterError_t::CHI2_ERROR:
      return 2;
    case RawFitterError_t::BUNCH_NOT_OK:
      return 3;
    case RawFitterError_t::LOW_SIGNAL:
      return 4;
  };
  // Silence compiler warnings for false positives
  // can never enter here due to usage of enum class
  return -1;
}

CaloRawFitter::RawFitterError_t CaloRawFitter::intToErrorType(unsigned int fiterror)
{
  assert(fiterror < getNumberOfErrorTypes());
  switch (fiterror) {
    case 0:
      return CaloRawFitter::RawFitterError_t::SAMPLE_UNINITIALIZED;
    case 1:
      return CaloRawFitter::RawFitterError_t::FIT_ERROR;
    case 2:
      return CaloRawFitter::RawFitterError_t::CHI2_ERROR;
    case 3:
      return CaloRawFitter::RawFitterError_t::BUNCH_NOT_OK;
    case 4:
      return CaloRawFitter::RawFitterError_t::LOW_SIGNAL;
  };
  // Silence the compiler warning for false positives
  // Since we catch invalid codes via the assert we can
  // never reach here. Since it is an enum class we need
  // to choose one error code here. Pick the first error
  // code instead.
  return CaloRawFitter::RawFitterError_t::SAMPLE_UNINITIALIZED;
}

const char* CaloRawFitter::getErrorTypeName(CaloRawFitter::RawFitterError_t fiterror)
{
  switch (fiterror) {
    case CaloRawFitter::RawFitterError_t::SAMPLE_UNINITIALIZED:
      return "SampleUninitalized";
    case CaloRawFitter::RawFitterError_t::FIT_ERROR:
      return "NoConvergence";
    case CaloRawFitter::RawFitterError_t::CHI2_ERROR:
      return "Chi2Error";
    case CaloRawFitter::RawFitterError_t::BUNCH_NOT_OK:
      return "BunchRejected";
    case CaloRawFitter::RawFitterError_t::LOW_SIGNAL:
      return "LowSignal";
  };
  // Silence compiler warnings for false positives
  // can never enter here due to usage of enum class
  return "Unknown";
}

const char* CaloRawFitter::getErrorTypeTitle(CaloRawFitter::RawFitterError_t fiterror)
{
  switch (fiterror) {
    case CaloRawFitter::RawFitterError_t::SAMPLE_UNINITIALIZED:
      return "sample uninitalized";
    case CaloRawFitter::RawFitterError_t::FIT_ERROR:
      return "No convergence";
    case CaloRawFitter::RawFitterError_t::CHI2_ERROR:
      return "Chi2 error";
    case CaloRawFitter::RawFitterError_t::BUNCH_NOT_OK:
      return "Bunch rejected";
    case CaloRawFitter::RawFitterError_t::LOW_SIGNAL:
      return "Low signal";
  };
  // Silence compiler warnings for false positives
  // can never enter here due to usage of enum class
  return "Unknown";
}

const char* CaloRawFitter::getErrorTypeDescription(CaloRawFitter::RawFitterError_t fiterror)
{
  switch (fiterror) {
    case CaloRawFitter::RawFitterError_t::SAMPLE_UNINITIALIZED:
      return "Sample for fit not initialzied or bunch length is 0";
    case CaloRawFitter::RawFitterError_t::FIT_ERROR:
      return "Fit of the raw bunch was not successful";
    case CaloRawFitter::RawFitterError_t::CHI2_ERROR:
      return "Chi2 of the fit could not be determined";
    case CaloRawFitter::RawFitterError_t::BUNCH_NOT_OK:
      return "Calo bunch could not be selected";
    case CaloRawFitter::RawFitterError_t::LOW_SIGNAL:
      return "No ADC value above threshold found";
  };
  // Silence compiler warnings for false positives
  // can never enter here due to usage of enum class
  return "Unknown error code";
}

// Default constructor
CaloRawFitter::CaloRawFitter(const char* name, const char* nameshort) : mMinTimeIndex(-1),
                                                                        mMaxTimeIndex(-1),
                                                                        mAmpCut(4),
                                                                        mNsamplePed(3),
                                                                        mIsZerosupressed(true),
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
    LOG(warn) << "Attempt to set Invalid time bin range (Min , Max) = (" << min << ", " << max << "), Ingored";
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

std::tuple<float, std::array<double, constants::EMCAL_MAXTIMEBINS>> CaloRawFitter::reverseAndSubtractPed(const Bunch& bunch) const
{
  std::array<double, constants::EMCAL_MAXTIMEBINS> outarray;
  int length = bunch.getBunchLength();
  const gsl::span<const uint16_t> sig(bunch.getADC());

  double ped = mIsZerosupressed ? 0. : evaluatePedestal(sig, length);

  for (int i = 0; i < length; i++) {
    outarray[i] = sig[length - i - 1] - ped;
  }

  return std::make_tuple(ped, outarray);
}

double CaloRawFitter::evaluatePedestal(const gsl::span<const uint16_t> data, std::optional<int> length) const
{
  if (!mNsamplePed) {
    throw RawFitterError_t::SAMPLE_UNINITIALIZED;
  }
  if (data.size() < mNsamplePed) {
    return 0.;
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

std::tuple<short, short, short> CaloRawFitter::selectMaximumBunch(const gsl::span<const Bunch>& bunchvector)
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
      throw RawFitterError_t::BUNCH_NOT_OK;
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

double CaloRawFitter::calculateChi2(double amp, double time,
                                    int first, int last,
                                    double adcErr, double tau) const
{
  if (first == last || first < 0) { // signal consists of single sample, chi2 estimate (0) not too well defined..
                                    // or, first is negative, the indices are not valid
    throw RawFitterError_t::CHI2_ERROR;
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
std::tuple<int, int, float, short, short, float, int, int> CaloRawFitter::preFitEvaluateSamples(const gsl::span<const Bunch> bunchvector, int adcThreshold)
{

  int nsamples(0), first(0), last(0), indexMaxADCRReveresed(0);
  double peakADC(0.), pedestal(0.);

  // Reset buffer for reversed bunch, no matter whether the bunch could be selected or not
  mReversed.fill(0);

  // select the bunch with the highest amplitude unless any time constraints is set
  for (unsigned int i = 0; i < bunchvector.size(); i++) {
    if (bunchvector[i].getBunchLength() > bunchvector[i].getADC().size()) {
      throw RawFitterError_t::FIT_ERROR;
    }
  }
  auto [bunchindex, indexMaxADC, adcMAX] = selectMaximumBunch(bunchvector);

  // something valid was found, and non-zero amplitude
  if (bunchindex >= 0) {
    if (adcMAX >= adcThreshold) {
      // use more convenient numbering and possibly subtract pedestal

      int bunchlength = bunchvector[bunchindex].getBunchLength();
      const std::vector<uint16_t>& sig = bunchvector[bunchindex].getADC();

      if (!mIsZerosupressed) {
        pedestal = evaluatePedestal(sig, bunchlength);
      }

      int testindexReverse = -1;
      for (int i = 0; i < bunchlength; i++) {
        mReversed[i] = sig[bunchlength - i - 1] - pedestal;
        if (mReversed[i] > peakADC) {
          peakADC = mReversed[i];
          testindexReverse = i;
        }
      }

      if (peakADC >= adcThreshold) // possibly significant signal
      {
        // select array around max to possibly be used in fit
        indexMaxADCRReveresed = indexMaxADC - bunchvector[bunchindex].getStartTime();
        std::tie(first, last) = findPeakRegion(gsl::span<double>(&mReversed[0], bunchvector[bunchindex].getBunchLength()), indexMaxADCRReveresed, adcThreshold);

        // sanity check: maximum should not be in first or last bin
        // if we should do a fit
        if (first != indexMaxADCRReveresed && last != indexMaxADCRReveresed) {
          // calculate how many samples we have
          nsamples = last - first + 1;
        }
      }
    } else {
      throw RawFitterError_t::LOW_SIGNAL;
    }
  } else {
    throw RawFitterError_t::BUNCH_NOT_OK;
  }

  return std::make_tuple(nsamples, bunchindex, peakADC, adcMAX, indexMaxADCRReveresed, pedestal, first, last);
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const CaloRawFitter::RawFitterError_t error)
{
  stream << CaloRawFitter::getErrorTypeName(error);
  return stream;
}
