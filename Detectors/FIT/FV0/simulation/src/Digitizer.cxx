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

ClassImp(o2::fv0::Digitizer);

using namespace o2::math_utils;
using namespace o2::fv0;

void Digitizer::clear()
{
  mEventId = -1;
  mSrcId = -1;
  mTimeStamp = 0;
  mMCLabels.clear();
  for (auto& analogSignal : mPmtChargeVsTime) {
    std::fill_n(std::begin(analogSignal), analogSignal.size(), 0);
  }
}

//_______________________________________________________________________
void Digitizer::init()
{
  LOG(INFO) << "V0Digitizer::init -> start = ";
  mTimeStamp = 0;

  mNBins = 2000;           //Will be computed using detector set-up from CDB
  mBinSize = 25.0 / 256.0; //Will be set-up from CDB

  for (Int_t i = 0; i < DP::NCHANNELS; i++) {
    mPmtChargeVsTime[i].resize(mNBins);
  }

  auto const roundVc = [&](int i) -> int {
    return (i / Vc::float_v::Size) * Vc::float_v::Size;
  };
  // set up PMT response tables
  Float_t offset = -0.5f * mBinSize; // offset \in [-0.5..0.5] * mBinSize
  Int_t const nBins = roundVc(std::lround(4.0f * DP::mPmtTransitTime / mBinSize));
  for (auto& table : mPmtResponseTables) {
    table.resize(nBins);
    Float_t t = -2.0f * DP::mPmtTransitTime + offset; // t \in offset + [-2 2] * DP::mPmtTransitTime
    for (Int_t j = 0; j < nBins; ++j) {
      table[j] = Digitizer::PmtResponse(t);
      t += mBinSize;
    }
    offset += mBinSize / Float_t(DP::NUM_PMT_RESPONSE_TABLES - 1);
  }

  TF1 scintDelayFn("fScintDelay", "gaus",
                   -6.0f * DP::mIntrinsicTimeRes,
                   +6.0f * DP::mIntrinsicTimeRes);
  scintDelayFn.SetParameters(1, 0, DP::mIntrinsicTimeRes);
  mRndScintDelay.initialize(scintDelayFn);

  // Initialize function describing the PMT time response
  TF1 pmtResponseFn("mPmtResponse",
                    &Digitizer::PmtResponse,
                    -1.0f * DP::mPmtTransitTime,
                    +2.0f * DP::mPmtTransitTime, 0);
  pmtResponseFn.SetNpx(100);
  mPmtTimeIntegral = pmtResponseFn.Integral(-1.0f * DP::mPmtTransitTime,
                                            +2.0f * DP::mPmtTransitTime);

  // Initialize function describing PMT response to the single photoelectron
  TF1 singlePhESpectrumFn("mSinglePhESpectrum",
                          &Digitizer::SinglePhESpectrum,
                          DP::photoelMin,
                          DP::photoelMax, 0);
  Float_t const meansPhE = singlePhESpectrumFn.Mean(DP::photoelMin, DP::photoelMax);
  mRndGainVar.initialize([&]() -> float {
    return singlePhESpectrumFn.GetRandom(DP::photoelMin, DP::photoelMax) / meansPhE;
  });

  TF1 signalShapeFn("signalShape", "crystalball", 0, 300);
  signalShapeFn.SetParameters(1,
                              DP::mShapeSigma,
                              DP::mShapeSigma,
                              DP::mShapeAlpha,
                              DP::mShapeN);
  mRndSignalShape.initialize([&]() -> float {
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
  std::iota(std::begin(hitIdx), std::end(hitIdx), 0);
  std::sort(std::begin(hitIdx), std::end(hitIdx), [&hits](int a, int b) { return hits[a].GetTrackID() < hits[b].GetTrackID(); });

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
    if (hitEdep < DP::singleMipThreshold) {
      continue;
    }

    Double_t const nPhotons = hitEdep * DP::N_PHOTONS_PER_MEV;
    Int_t const nPhE = SimulateLightYield(detId, nPhotons);
    Float_t const t = mRndScintDelay.getNextValue() + hit.GetTime() * 1e9;
    Float_t const charge = TMath::Qe() * DP::mPmtGain * mBinSize / mPmtTimeIntegral;

    auto& analogSignal = mPmtChargeVsTime[detId];

    for (Int_t iPhE = 0; iPhE < nPhE; ++iPhE) {
      Float_t const tPhE = t + mRndSignalShape.getNextValue();
      Int_t const firstBin = roundVc(TMath::Max((Int_t)0, (Int_t)((tPhE - DP::mPmtTransitTime) / mBinSize)));
      Int_t const lastBin = TMath::Min((Int_t)mNBins - 1, (Int_t)((tPhE + 2. * DP::mPmtTransitTime) / mBinSize));
      Float_t const tempT = mBinSize * (0.5f + firstBin) - tPhE;
      Float_t* p = analogSignal.data() + firstBin;
      long iStart = std::lround((tempT + 2.0f * DP::mPmtTransitTime) / mBinSize);
      float const offset = tempT + 2.0f * DP::mPmtTransitTime - Float_t(iStart) * mBinSize;
      long const iOffset = std::lround(offset / mBinSize * Float_t(DP::NUM_PMT_RESPONSE_TABLES - 1));
      if (iStart < 0) { // this should not happen
        LOG(ERROR) << "V0Digitizer: table lookup failure";
      }
      iStart = roundVc(std::max(long(0), iStart));

      Vc::float_v workVc;
      Vc::float_v pmtVc;
      Float_t const* q = mPmtResponseTables[DP::NUM_PMT_RESPONSE_TABLES / 2 + iOffset].data() + iStart;
      Float_t const* qEnd = &mPmtResponseTables[DP::NUM_PMT_RESPONSE_TABLES / 2 + iOffset].back();
      for (Int_t i = firstBin, iEnd = roundVc(lastBin); q < qEnd && i < iEnd; i += Vc::float_v::Size) {
        pmtVc.load(q);
        q += Vc::float_v::Size;
        Vc::prefetchForOneRead(q);
        workVc.load(p);
        workVc += mRndGainVar.getNextValueVc() * charge * pmtVc;
        workVc.store(p);
        p += Vc::float_v::Size;
        Vc::prefetchForOneRead(p);
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
  size_t const first = digitsCh.size();
  size_t nStored = 0;
  for (Int_t ipmt = 0; ipmt < DP::NCHANNELS; ++ipmt) {
    Float_t totalCharge = 0.0f;
    auto const& analogSignal = mPmtChargeVsTime[ipmt];
    for (Int_t iTimeBin = 0; iTimeBin < mNBins; ++iTimeBin) {
      Float_t const timeBinCharge = mPmtChargeVsTime[ipmt][iTimeBin];
      totalCharge += timeBinCharge;
    }
    totalCharge *= DP::INV_CHARGE_PER_ADC;
    digitsCh.emplace_back(ipmt, SimulateTimeCfd(ipmt), std::lround(totalCharge));
    ++nStored;
  }

  // Send MClabels and digitsBC to storage
  size_t const nBC = digitsBC.size();
  digitsBC.emplace_back(first, nStored, mIntRecord);
  for (auto const& lbl : mMCLabels) {
    labels.addElement(nBC, lbl);
  }
}

// -------------------------------------------------------------------------------
// --- Internal helper methods related to conversion of energy-deposition into ---
// --- photons -> photoelectrons -> electrical signal                          ---
// -------------------------------------------------------------------------------
Int_t Digitizer::SimulateLightYield(Int_t pmt, Int_t nPhot) const
{
  const Float_t epsilon = 0.0001f;
  const Float_t p = DP::mLightYield * DP::mPhotoCathodeEfficiency;
  if ((fabs(1.0f - p) < epsilon) || nPhot == 0) {
    return nPhot;
  }
  const Int_t n = Int_t(nPhot < 100
                          ? gRandom->Binomial(nPhot, p)
                          : gRandom->Gaus(p * nPhot + 0.5, TMath::Sqrt(p * (1 - p) * nPhot)));
  return n;
}

//_______________________________________________________________________
Float_t Digitizer::SimulateTimeCfd(Int_t channel) const
{
  Float_t timeCfd = -1024.0f;
  Int_t const binShift = TMath::Nint(DP::mTimeShiftCfd / mBinSize);
  Float_t sigPrev = -mPmtChargeVsTime[channel][0];
  for (Int_t iTimeBin = 1; iTimeBin < mNBins; ++iTimeBin) {
    Float_t const sigCurrent = (iTimeBin >= binShift
                                  ? 5.0f * mPmtChargeVsTime[channel][iTimeBin - binShift] - mPmtChargeVsTime[channel][iTimeBin]
                                  : -mPmtChargeVsTime[channel][iTimeBin]);
    if (sigPrev < 0.0f && sigCurrent >= 0.0f) {
      timeCfd = Float_t(iTimeBin) * mBinSize;
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
  if (x > 2 * DP::mPmtTransitTime)
    return 0.0;
  if (x < -DP::mPmtTransitTime)
    return 0.0;
  x += DP::mPmtTransitTime;
  Double_t const x2 = x * x;
  return x2 * std::exp(-x2 * DP::mOneOverPmtTransitTime2);
}

//_______________________________________________________________________
Double_t Digitizer::SinglePhESpectrum(Double_t* x, Double_t*)
{
  // x -- number of photo-electrons emitted from the first dynode
  // this function describes the PMT amplitude response to a single photoelectron
  if (x[0] < 0.0)
    return 0.0;
  return (TMath::Poisson(x[0], DP::mPmtNbOfSecElec) +
          DP::mPmtTransparency * TMath::Poisson(x[0], 1.0));
}
