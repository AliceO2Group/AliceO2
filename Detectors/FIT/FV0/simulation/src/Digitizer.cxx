// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FV0Simulation/Digitizer.h"

#include <TMath.h>
#include <TRandom.h>
#include <TH1F.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <optional>
#include <MathUtils/CachingTF1.h>
#include <boost/format.hpp>

using namespace o2::math_utils;
using namespace o2::fv0;
using boost::format;

ClassImp(Digitizer);

void Digitizer::clear()
{
  mEventId = -1;
  mSrcId = -1;
  mMCLabels.clear();
  for (auto analogSignalPulse : mPmtChargeVsTime) {
    analogSignalPulse.clear();
  }
}

//_______________________________________________________________________
void Digitizer::init()
{
  LOG(INFO) << "V0Digitizer::init -> start = ";

  // Zero iterators for random rings
  mIteratorScintDelay = 0;
  mIteratorGainVar = 0;
  mIteratorSignalShape = 0;

  const float y = DigitizationParameters::mPmtTransitTime;
  format pmresponse("(x+%1%)*(x+%2%)*TMath::Exp(-(x+%3%)*(x+%4%)/(%5% * %6%))");
  pmresponse % y % y % y % y % y % y;
  std::string pmtResponseFormula = pmresponse.str();

  mNBins = 2000;           //Will be computed using detector set-up from CDB
  mBinSize = 25.0 / 256.0; //Will be set-up from CDB

  for (Int_t i = 0; i < DigitizationParameters::NCHANNELS; i++) {
    mPmtChargeVsTime[i].resize(mNBins);
  }

  // Initialize function describing the PMT time response
  std::unique_ptr<o2::base::CachingTF1> mPmtResponse = std::make_unique<o2::base::CachingTF1>("mPmtResponse", pmtResponseFormula.c_str(), -y, 2 * y);
  mPmtResponse->SetNpx(100);
  mPmtTimeIntegral = mPmtResponse->Integral(-DigitizationParameters::mPmtTransitTime,
                                            2. * DigitizationParameters::mPmtTransitTime);

  // Prefill random ring with time delay from scintillator
  for (int i = 0; i < DigitizationParameters::HIT_RANDOM_RING_SIZE; i++) {
    mScintillatorDelay.at(i) = gRandom->Gaus(0, DigitizationParameters::mIntrinsicTimeRes);
  }

  // Initialize function describing PMT response to the single photoelectron
  std::unique_ptr<o2::base::CachingTF1> mSinglePhESpectrum = std::make_unique<o2::base::CachingTF1>("mSinglePhESpectrum", this, &Digitizer::SinglePhESpectrum, DigitizationParameters::photoelMin, DigitizationParameters::photoelMax, 0);
  Float_t meansPhE = mSinglePhESpectrum->Mean(DigitizationParameters::photoelMin, DigitizationParameters::photoelMax);
  // Prefill random ring with variation of gain with respect to its mean value
  for (int i = 0; i < DigitizationParameters::PHE_RANDOM_RING_SIZE; i++) {
    mGainVar.at(i) = mSinglePhESpectrum->GetRandom(DigitizationParameters::photoelMin, DigitizationParameters::photoelMax) / meansPhE;
  }

  std::unique_ptr<o2::base::CachingTF1> signalShape;
  signalShape = std::make_unique<o2::base::CachingTF1>("signalShape", "crystalball", 0, 300);
  signalShape->SetParameters(1,
                             DigitizationParameters::mShapeSigma,
                             DigitizationParameters::mShapeSigma,
                             DigitizationParameters::mShapeAlpha,
                             DigitizationParameters::mShapeN);
  // Prefill random ring with time delay of photoelectron hit coming from measured signal shape
  for (int i = 0; i < DigitizationParameters::PHE_RANDOM_RING_SIZE; i++) {
    mSignalShape.at(i) = signalShape->GetRandom(0, mBinSize * Float_t(mNBins));
  }

  LOG(INFO) << "V0Digitizer::init -> finished";
}

void Digitizer::process(const std::vector<o2::fv0::Hit>& hits,
                        std::vector<o2::fv0::BCData>& digitsBC,
                        std::vector<o2::fv0::ChannelData>& digitsCh,
                        o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>& labels)
{
  LOG(INFO) << "[FV0] Digitizer::process(): begin with " << hits.size() << " hits";

  auto sorted_hits{hits};
  std::sort(sorted_hits.begin(), sorted_hits.end(), [](o2::fv0::Hit const& a, o2::fv0::Hit const& b) {
    return a.GetTrackID() < b.GetTrackID();
  });

  Int_t parentIdPrev = -10;
  for (auto& hit : sorted_hits) {
    int detId = hit.GetDetectorID();
    const double hitEdep = hit.GetHitValue() * 1e3; //convert to MeV

    // TODO: check how big is inaccuracy if more than 1 'below-threshold' particles hit the same detector cell
    if (hitEdep < DigitizationParameters::singleMipThreshold) {
      continue;
    }

    Double_t const nPhotons = hitEdep * DigitizationParameters::N_PHOTONS_PER_MEV;
    Int_t const nPhE = SimulateLightYield(detId, nPhotons);
    Float_t const t = mScintillatorDelay.at(mIteratorScintDelay) + hit.GetTime() * 1e9;
    Float_t charge = TMath::Qe() * DigitizationParameters::mPmtGain * mBinSize / mPmtTimeIntegral;
    for (Int_t iPhE = 0; iPhE < nPhE; ++iPhE) {
      Float_t const tPhE = t + mSignalShape.at(mIteratorSignalShape);
      Int_t const firstBin = TMath::Max((Int_t)0, (Int_t)((tPhE - DigitizationParameters::mPmtTransitTime) / mBinSize));
      Int_t const lastBin = TMath::Min((Int_t)mNBins - 1, (Int_t)((tPhE + 2. * DigitizationParameters::mPmtTransitTime) / mBinSize));
      for (Int_t iBin = firstBin; iBin <= lastBin; ++iBin) {
        Float_t const tempT = mBinSize * (0.5 + iBin) - tPhE;
        mPmtChargeVsTime.at(detId).at(iBin) += mGainVar.at(mIteratorGainVar) * charge * PmtResponse(tempT);
      } // time-bin loop

      // RandomRing servicing (perhaps a separate class could do better)
      mIteratorGainVar++;
      if (mIteratorGainVar >= mGainVar.size()) {
        mIteratorGainVar = 0;
      }
      mIteratorSignalShape++;
      if (mIteratorSignalShape >= mSignalShape.size()) {
        mIteratorSignalShape = 0;
      }
    }   //photo electron loop
    mIteratorScintDelay++;
    if (mIteratorScintDelay >= mScintillatorDelay.size()) {
      mIteratorScintDelay = 0;
    }

    // Charged particles in MCLabel
    Int_t parentId = hit.GetTrackID();
    if (parentId != parentIdPrev) {
      mMCLabels.emplace_back(parentId, mEventId, mSrcId, detId);
      parentIdPrev = parentId;
    }
  } //hit loop

  // Sum charge of all time bins to get total charge collected for a given channel
  int first = digitsCh.size(), nStored = 0;
  for (Int_t ipmt = 0; ipmt < DigitizationParameters::NCHANNELS; ++ipmt) {
    Float_t totalCharge = 0;
    for (Int_t iTimeBin = 0; iTimeBin < mNBins; ++iTimeBin) {
      Float_t timeBinCharge = mPmtChargeVsTime[ipmt][iTimeBin] / DigitizationParameters::CHARGE_PER_ADC;
      totalCharge += timeBinCharge;
    }
    digitsCh.emplace_back(ipmt, SimulateTimeCfd(ipmt), totalCharge);
    nStored++;
  }

  // Send MClabels and digitsBC to storage
  int nBC = digitsBC.size();
  digitsBC.emplace_back(first, nStored, mIntRecord);
  for (const auto& lbl : mMCLabels) {
    labels.addElement(nBC, lbl);
  }
}

// -------------------------------------------------------------------------------
// --- Internal helper methods related to conversion of energy-deposition into ---
// --- photons -> photoelectrons -> electrical signal                          ---
// -------------------------------------------------------------------------------
Int_t Digitizer::SimulateLightYield(Int_t pmt, Int_t nPhot)
{
  const Float_t epsilon = 0.0001;
  const Float_t p = DigitizationParameters::mLightYield * DigitizationParameters::mPhotoCathodeEfficiency;
  if ((fabs(1.0f - p) < epsilon) || nPhot == 0) {
    return nPhot;
  }
  const Int_t n = Int_t(nPhot < 100
                          ? gRandom->Binomial(nPhot, p)
                          : gRandom->Gaus(p * nPhot + 0.5, TMath::Sqrt(p * (1 - p) * nPhot)));
  return n;
}

//_______________________________________________________________________
Float_t Digitizer::SimulateTimeCfd(Int_t channel)
{
  //std::fill(mPmtChargeVsTimeCfd.begin(), mPmtChargeVsTimeCfd.end(), 0);
  Float_t timeCfd = -1024;
  Int_t const binShift = TMath::Nint(DigitizationParameters::mTimeShiftCfd / mBinSize);
  Float_t sigPrev = -mPmtChargeVsTime[channel][0];
  for (Int_t iTimeBin = 1; iTimeBin < mNBins; ++iTimeBin) {
    //if (mPmtChargeVsTime[channel][iTimeBin] != 0) std::cout << mPmtChargeVsTime[channel][iTimeBin] / parameters.CHARGE_PER_ADC << ", ";
    Float_t sigCurrent = (iTimeBin >= binShift
                            ? 5.0 * mPmtChargeVsTime[channel][iTimeBin - binShift] - mPmtChargeVsTime[channel][iTimeBin]
                            : -mPmtChargeVsTime[channel][iTimeBin]);
    if (sigPrev < 0 && sigCurrent >= 0) {
      timeCfd = mBinSize * Float_t(iTimeBin);
      //std::cout<<timeCfd<<std::endl;
      break;
    }
    sigPrev = sigCurrent;
  }
  return timeCfd;
}

//_______________________________________________________________________
Float_t Digitizer::PmtResponse(Float_t x)
{
  // this function describes the PMT time response to a single photoelectron
  x += DigitizationParameters::mPmtTransitTime;
  Float_t x2 = x * x;
  return x2 * TMath::Exp(-x2 / DigitizationParameters::mPmtTransitTime2);
}

//_______________________________________________________________________
Double_t Digitizer::SinglePhESpectrum(Double_t* x, Double_t*)
{
  // x -- number of photo-electrons emitted from the first dynode
  // this function describes the PMT amplitude response to a single photoelectron
  if (x[0] < 0)
    return 0;
  return (TMath::Poisson(x[0], DigitizationParameters::mPmtNbOfSecElec) +
          DigitizationParameters::mPmtTransparency * TMath::Poisson(x[0], 1.0));
}
