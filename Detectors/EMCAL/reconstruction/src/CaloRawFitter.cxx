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
/// \author Hadi Hassan (hadi.hassan@cern.ch)

#include "FairLogger.h"
#include <gsl/span>

// ROOT sytem
#include "TMath.h"

#include "EMCALReconstruction/Bunch.h"
#include "EMCALReconstruction/CaloFitResults.h"
#include "DataFormatsEMCAL/Constants.h"

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

unsigned short CaloRawFitter::maxAmp(const gsl::span<unsigned short> data) const
{
  return *std::max_element(data.begin(), data.end());
}

std::tuple<int, int> CaloRawFitter::selectSubarray(const gsl::span<double> data, short maxindex, int cut) const
{

  int first(0), last(0);
  int tmpfirst = maxindex;
  int tmplast = maxindex;
  double prevFirst = data[maxindex];
  double prevLast = data[maxindex];
  bool firstJump = false;
  bool lastJump = false;

  while ((tmpfirst >= 0) && (data[tmpfirst] >= cut) && (!firstJump)) {
    // jump check:
    if (tmpfirst != maxindex) { // neighbor to maxindex can share peak with maxindex
      if (data[tmpfirst] >= prevFirst) {
        firstJump = true;
      }
    }
    prevFirst = data[tmpfirst];
    tmpfirst--;
  }

  while ((tmplast < data.size()) && (data[tmplast] >= cut) && (!lastJump)) {
    // jump check:
    if (tmplast != maxindex) { // neighbor to maxindex can share peak with maxindex
      if (data[tmplast] >= prevLast) {
        lastJump = true;
      }
    }
    prevLast = data[tmplast];
    tmplast++;
  }

  // we keep one pre- or post- sample if we can (as in online)
  // check though if we ended up on a 'jump', or out of bounds: if so, back up
  if (firstJump || tmpfirst < 0)
    tmpfirst++;
  if (lastJump || tmplast >= data.size())
    tmplast--;

  first = tmpfirst;
  last = tmplast;

  return std::make_tuple(first, last);
}

std::tuple<float, std::array<double, constants::EMCAL_MAXTIMEBINS>> CaloRawFitter::reverseAndSubtractPed(const Bunch& bunch,
                                                                                                         std::optional<unsigned int> altrocfg1, std::optional<unsigned int> altrocfg2) const
{
  std::array<double, constants::EMCAL_MAXTIMEBINS> outarray;
  int length = bunch.getBunchLength();
  const std::vector<uint16_t>& sig = bunch.getADC();

  double ped = evaluatePedestal(sig, length);

  for (int i = 0; i < length; i++) {
    outarray[i] = sig[length - i - 1] - ped;
  }

  return std::make_tuple(ped, outarray);
}

float CaloRawFitter::evaluatePedestal(const std::vector<uint16_t>& data, std::optional<int> length) const
{
  double tmp = 0;

  if (mIsZerosupressed == false) {
    for (int i = 0; i < mNsamplePed; i++) {
      tmp += data[i];
    }
  }

  return tmp / mNsamplePed;
}

short CaloRawFitter::maxAmp(const Bunch& bunch, int& maxindex) const
{
  short tmpmax = -1;
  int tmpindex = -1;
  const std::vector<uint16_t>& sig = bunch.getADC();

  for (int i = 0; i < bunch.getBunchLength(); i++) {
    if (sig[i] > tmpmax) {
      tmpmax = sig[i];
      tmpindex = i;
    }
  }

  if (maxindex != 0) {
    maxindex = bunch.getBunchLength() - 1 - tmpindex + bunch.getStartTime();
  }

  return tmpmax;
}

bool CaloRawFitter::checkBunchEdgesForMax(const Bunch& bunch) const
{
  short tmpmax = -1;
  int tmpindex = -1;
  const std::vector<uint16_t>& sig = bunch.getADC();

  for (int i = 0; i < bunch.getBunchLength(); i++) {
    if (sig[i] > tmpmax) {
      tmpmax = sig[i];
      tmpindex = i;
    }
  }

  bool bunchOK = true;
  if (tmpindex == 0 || tmpindex == (bunch.getBunchLength() - 1)) {
    bunchOK = false;
  }

  return bunchOK;
}

std::tuple<short, short, short> CaloRawFitter::selectBunch(const std::vector<Bunch>& bunchvector)
{
  short bunchindex = -1;
  short maxall = -1;
  int indx = -1;
  short maxampbin(0), maxamplitude(0);

  for (unsigned int i = 0; i < bunchvector.size(); i++) {
    short max = maxAmp(bunchvector.at(i), indx); // CRAP PTH, bug fix, trouble if more than one bunches
    if (isInTimeRange(indx, mMaxTimeIndex, mMinTimeIndex)) {
      if (max > maxall) {
        maxall = max;
        bunchindex = i;
        maxampbin = indx;
        maxamplitude = max;
      }
    }
  }

  if (bunchindex >= 0) {
    bool bunchOK = checkBunchEdgesForMax(bunchvector.at(bunchindex));
    if (!bunchOK) {
      bunchindex = -1;
    }
  }

  return std::make_tuple(bunchindex, maxampbin, maxamplitude);
}

bool CaloRawFitter::isInTimeRange(int maxindex, int maxtindx, int mintindx) const
{
  if ((mintindx < 0 && maxtindx < 0) || maxtindx < 0) {
    return true;
  }

  return (maxindex < maxtindx) && (maxindex > mintindx) ? true : false;
}

double CaloRawFitter::calculateChi2(double amp, double time,
                                    int first, int last,
                                    double adcErr, double tau) const
{
  if (first == last || first < 0) { // signal consists of single sample, chi2 estimate (0) not too well defined..
                                    // or, first is negative, the indices are not valid
    return -1;
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
std::tuple<int, int, float, short, short, float, int, int> CaloRawFitter::preFitEvaluateSamples(const std::vector<Bunch>& bunchvector,
                                                                                                std::optional<unsigned int> altrocfg1, std::optional<unsigned int> altrocfg2, int acut)
{

  float maxf(0), ped(0);
  int nsamples(0), first(0), last(0);
  short maxrev(0);

  // select the bunch with the highest amplitude unless any time constraints is set
  auto [index, maxampindex, maxamp] = selectBunch(bunchvector);

  // something valid was found, and non-zero amplitude
  if (index >= 0 && maxamp >= acut) {
    // use more convenient numbering and possibly subtract pedestal

    //std::tie(ped, mReversed) = reverseAndSubtractPed((bunchvector.at(index)), altrocfg1, altrocfg2);
    //maxf = (float)*std::max_element(mReversed.begin(), mReversed.end());

    int length = bunchvector.at(index).getBunchLength();
    const std::vector<uint16_t>& sig = bunchvector.at(index).getADC();

    double ped = evaluatePedestal(sig, length);

    for (int i = 0; i < length; i++) {
      mReversed[i] = sig[length - i - 1] - ped;
      if (maxf < mReversed[i])
        maxf = mReversed[i];
    }

    if (maxf >= acut) // possibly significant signal
    {
      // select array around max to possibly be used in fit
      maxrev = maxampindex - bunchvector.at(index).getStartTime();
      std::tie(first, last) = selectSubarray(gsl::span<double>(&mReversed[0], bunchvector.at(index).getBunchLength()), maxrev, acut);

      // sanity check: maximum should not be in first or last bin
      // if we should do a fit
      if (first != maxrev && last != maxrev) {
        // calculate how many samples we have
        nsamples = last - first + 1;
      }
    }
  }

  return std::make_tuple(nsamples, index, maxf, maxamp, maxrev, ped, first, last);
}
