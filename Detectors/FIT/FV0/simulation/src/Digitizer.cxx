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

#include <TRandom.h>
#include <algorithm>

using namespace o2::math_utils;
using namespace o2::fv0;

ClassImp(Digitizer);

void Digitizer::clear()
{
  LOG(INFO) << "V0Digitizer::clear ";
  mEventId = -1;
  mSrcId = -1;
  mTimeStamp = 0;
  mMCLabels.clear();

  for (auto& analogSignal : mPmtChargeVsTime)
    std::fill_n(std::begin(analogSignal), analogSignal.size(), 0);
}

//_______________________________________________________________________
void Digitizer::init()
{
  LOG(INFO) << "V0Digitizer::init -> start = ";
  mTimeStamp = 0;

  const float y = DigitizationParameters::mPmtTransitTime;

  mNBins = 2000;           //Will be computed using detector set-up from CDB
  mBinSize = 25.0 / 256.0; //Will be set-up from CDB

  for (Int_t i = 0; i < DigitizationParameters::NCHANNELS; i++)
    mPmtChargeVsTime[i].resize(mNBins);

  // Initialize function describing the PMT time response
  TF1 pmtResponseFn("mPmtResponse",
                    &Digitizer::PmtResponse,
                    -y, 2 * y, 0);
  pmtResponseFn.SetNpx(100);
  mPmtTimeIntegral = pmtResponseFn.Integral(-y, 2. * y);

  mRndScintDelay.initialize([&]() {
    return gRandom->Gaus(0, DigitizationParameters::mIntrinsicTimeRes);
  });

  // Initialize function describing PMT response to the single photoelectron
  TF1 singlePhESpectrumFn("mSinglePhESpectrum", this,
                          &Digitizer::SinglePhESpectrum,
                          DigitizationParameters::photoelMin,
                          DigitizationParameters::photoelMax, 0);
  Float_t const meansPhE = singlePhESpectrumFn.Mean(DigitizationParameters::photoelMin, DigitizationParameters::photoelMax);
  mRndGainVar.initialize([&]() {
    return singlePhESpectrumFn.GetRandom(DigitizationParameters::photoelMin,
                                         DigitizationParameters::photoelMax) /
           meansPhE;
  });

  TF1 signalShapeFn("signalShape", "crystalball", 0, 300);
  signalShapeFn.SetParameters(1,
                              DigitizationParameters::mShapeSigma,
                              DigitizationParameters::mShapeSigma,
                              DigitizationParameters::mShapeAlpha,
                              DigitizationParameters::mShapeN);
  mRndSignalShape.initialize([&]() {
    return signalShapeFn.GetRandom(0, mBinSize * Float_t(mNBins));
  });

  LOG(INFO) << "V0Digitizer::init -> finished";
}

void Digitizer::process(const std::vector<o2::fv0::Hit>& hits,
                        std::vector<o2::fv0::BCData>& digitsBC,
                        std::vector<o2::fv0::ChannelData>& digitsCh,
                        o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>& labels)
{
  LOG(INFO) << "[FV0] Digitizer::process(): begin with " << hits.size() << " hits";

  std::vector<int> hitIdx(hits.size());
  std::iota(hitIdx.begin(), hitIdx.end(), 0);
  std::sort(hitIdx.begin(), hitIdx.end(), [&](int a, int b) { return hits[a].GetTrackID() < hits[b].GetTrackID(); });

  auto const PmtResponseVc = [&](Vc::float_v x) -> Vc::float_v {
    if (x[0] > 2 * DigitizationParameters::mPmtTransitTime) {
      return Vc::float_v(0);
    }
    if (x[Vc::float_v::Size - 1] < -DigitizationParameters::mPmtTransitTime) {
      return Vc::float_v(0);
    }
    x += Vc::float_v(DigitizationParameters::mPmtTransitTime);
    Vc::float_v const x2 = x * x;
    Vc::float_v const c = Vc::float_v(-1.0f * DigitizationParameters::mOneOverPmtTransitTime2);
    return x2 * Vc::exp(x2 * c);
  };

  auto const roundVc = [&](int i) -> int {
    return (i / Vc::float_v::Size) * Vc::float_v::Size;
  };
  Int_t parentIdPrev = -10;
  // use ordered hits
  for (auto ids : hitIdx) {
    const auto& hit = hits[ids];
    Int_t const detId = hit.GetDetectorID();
    const Double_t hitEdep = hit.GetHitValue() * 1e3; //convert to MeV

    // TODO: check how big is inaccuracy if more than 1 'below-threshold' particles hit the same detector cell
    if (hitEdep < DigitizationParameters::singleMipThreshold) {
      continue;
    }

    Double_t const nPhotons = hitEdep * DigitizationParameters::N_PHOTONS_PER_MEV;
    Int_t const nPhE = SimulateLightYield(detId, nPhotons);
    Float_t const t = mRndScintDelay.getNextValue() + hit.GetTime() * 1e9;
    Float_t const charge = TMath::Qe() * DigitizationParameters::mPmtGain * mBinSize / mPmtTimeIntegral;

    auto& analogSignal = mPmtChargeVsTime[detId];

    for (Int_t iPhE = 0; iPhE < nPhE; ++iPhE) {
      Float_t const tPhE = t + mRndSignalShape.getNextValue();
      Int_t const firstBin = roundVc(TMath::Max((Int_t)0, (Int_t)((tPhE - DigitizationParameters::mPmtTransitTime) / mBinSize)));
      Int_t const lastBin = TMath::Min((Int_t)mNBins - 1, (Int_t)((tPhE + 2. * DigitizationParameters::mPmtTransitTime) / mBinSize));
      Float_t tempT = mBinSize * (0.5f + firstBin) - tPhE;
      Float_t* p = analogSignal.data() + firstBin;
      Vc::float_v binVc(tempT);
      binVc += mBinSize * Vc::float_v::IndexesFromZero();
      Vc::float_v binIncVc(mBinSize * Vc::float_v::Size);

      Vc::float_v workVc;
      Int_t iBin = firstBin;
      for (Int_t const lastBinRounded = roundVc(lastBin); iBin < lastBinRounded;) {
        workVc.load(p);
        workVc += mRndGainVar.getNextValueVc() * charge * PmtResponseVc(binVc);
        workVc.store(p);
        Vc::prefetchForOneRead(p);
        binVc += binIncVc;
        p += Vc::float_v::Size;
        iBin += Vc::float_v::Size;
      }
      tempT += binVc[0] + mBinSize;
      // loop over the SIMD leftovers
      for (Float_t const* pEnd = analogSignal.data() + lastBin; p < pEnd; ++p) {
        *p += mRndGainVar.getNextValue() * charge * PmtResponse(tempT);
        tempT += mBinSize;
      }
    } //photo electron loop

    // Charged particles in MCLabel
    Int_t const parentId = hit.GetTrackID();
    if (parentId != parentIdPrev) {
      mMCLabels.emplace_back(parentId, mEventId, mSrcId, detId);
      parentIdPrev = parentId;
    }
  } //hit loop

  // Sum charge of all time bins to get total charge collected for a given channel
  Int_t first = digitsCh.size(), nStored = 0;
  for (Int_t ipmt = 0; ipmt < DigitizationParameters::NCHANNELS; ++ipmt) {
    Float_t totalCharge = 0.0f;
    for (Int_t iTimeBin = 0; iTimeBin < mNBins; ++iTimeBin) {
      Float_t const timeBinCharge = mPmtChargeVsTime[ipmt][iTimeBin];
      totalCharge += timeBinCharge;
    }
    totalCharge *= DigitizationParameters::INV_CHARGE_PER_ADC;
    digitsCh.emplace_back(ipmt, SimulateTimeCfd(ipmt), std::lround(totalCharge));
    nStored++;
  }

  // Send MClabels and digitsBC to storage
  Int_t nBC = digitsBC.size();
  digitsBC.emplace_back(first, nStored, mIntRecord);
  for (auto const& lbl : mMCLabels) {
    labels.addElement(nBC, lbl);
  }
}

// -------------------------------------------------------------------------------
// --- Internal helper methods related to conversion of energy-deposition into ---
// --- photons -> photoelectrons -> electrical signal                          ---
// -------------------------------------------------------------------------------
Int_t Digitizer::SimulateLightYield(Int_t pmt, Int_t nPhot)
{
  const Float_t epsilon = 0.0001f;
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

Double_t Digitizer::PmtResponse(Double_t* x, Double_t*)
{
  return Digitizer::PmtResponse(x[0]);
}
//_______________________________________________________________________
Double_t Digitizer::PmtResponse(Double_t x)
{
  // this function describes the PMT time response to a single photoelectron
  if (x > 2 * DigitizationParameters::mPmtTransitTime)
    return 0.0f;
  if (x < -DigitizationParameters::mPmtTransitTime)
    return 0.0f;
  x += DigitizationParameters::mPmtTransitTime;
  Double_t const x2 = x * x;
  return x2 * expf(-x2 * DigitizationParameters::mOneOverPmtTransitTime2);
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
