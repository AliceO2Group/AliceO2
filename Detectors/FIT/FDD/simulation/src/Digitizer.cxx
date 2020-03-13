// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FDDSimulation/Digitizer.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <CommonDataFormat/InteractionRecord.h>

#include "TMath.h"
#include "TRandom.h"
#include <algorithm>
#include <cassert>
#include <iostream>

using namespace o2::math_utils;
using namespace o2::fdd;

ClassImp(Digitizer);

Digitizer::BCCache::BCCache()
{
  memset(&pulse, 0, mNchannels * sizeof(ChannelBCDataF));
}

//_____________________________________________________________________________
void Digitizer::process(const std::vector<o2::fdd::Hit>& hits,
                        std::vector<o2::fdd::Digit>& digitsBC,
                        std::vector<o2::fdd::ChannelData>& digitsCh,
                        o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>& labels)
{
  // loop over all hits and produce digits
  //LOG(INFO) << "Processing IR = " << mIntRecord << " | NHits = " << hits.size();

  flush(digitsBC, digitsCh, labels); // flush cached signal which cannot be affect by new event

  auto sorted_hits{hits};
  std::sort(sorted_hits.begin(), sorted_hits.end(), [](o2::fdd::Hit const& a, o2::fdd::Hit const& b) {
    return a.GetTrackID() < b.GetTrackID();
  });
  LOG(INFO) << "Pulse";
  //Conversion of hits to the analogue pulse shape
  for (auto& hit : sorted_hits) {
    if (hit.GetTime() > 20e3) {
      const int maxWarn = 10;
      static int warnNo = 0;
      if (warnNo < maxWarn) {
        LOG(WARNING) << "Ignoring hit with time_in_event = " << hit.GetTime() << " ns"
                     << ((++warnNo < maxWarn) ? "" : " (suppressing further warnings)");
      }
      continue;
    }

    std::array<o2::InteractionRecord, NBC2Cache> cachedIR;
    Int_t pmt = hit.GetDetectorID();
    Int_t nPhE = SimulateLightYield(pmt, hit.GetNphot());

    Float_t dt_scintillator = mRndScintDelay.getNextValue();
    Float_t t = dt_scintillator + hit.GetTime();

    double hTime = t - getTOFCorrection(int(pmt / 4)); // account for TOF to detector
    hTime += mIntRecord.timeNS;
    o2::InteractionRecord irHit(hTime); // BC in which the hit appears (might be different from interaction BC for slow particles)

    // nominal time of the BC to which the hit will be attributed
    double bcTime = o2::InteractionRecord::bc2ns(irHit.bc, irHit.orbit);

    int nCachedIR = 0;
    for (int i = BCCacheMin; i < BCCacheMax + 1; i++) {
      double tNS = hTime + o2::constants::lhc::LHCBunchSpacingNS * i;
      cachedIR[nCachedIR].setFromNS(tNS);
      if (tNS < 0 && cachedIR[nCachedIR] > irHit) {
        continue; // don't go to negative BC/orbit (it will wrap)
      }
      setBCCache(cachedIR[nCachedIR++]); // ensure existence of cached container
    }
    // if digit for this sector does not exist, create one otherwise add to it
    Pulse(nPhE, hit.GetTrackID(), hTime, cachedIR, nCachedIR, pmt);

  } //hit loop
}

//_____________________________________________________________________________
void Digitizer::Pulse(int nPhE, int parID, double timeHit, std::array<o2::InteractionRecord, NBC2Cache> const& cachedIR, int nCachedIR, int channel)
{
  auto const roundVc = [&](int i) -> int {
    return (i / Vc::float_v::Size) * Vc::float_v::Size;
  };

  double time0 = cachedIR[0].bc2ns(); // start time of the 1st cashed BC
  float timeDiff = time0 - timeHit;
  if (channel < 9)
    timeDiff += parameters.mTimeDelayFDC;
  else
    timeDiff += parameters.mTimeDelayFDA;

  //LOG(INFO) <<"Ch = "<<channel<<" NphE = " << nPhE <<" timeDiff "<<timeDiff;
  Float_t charge = TMath::Qe() * parameters.mPmGain * mBinSize / mPmtTimeIntegral / mChargePerADC;

  Bool_t added[nCachedIR];
  for (int ir = 0; ir < nCachedIR; ir++)
    added[ir] = kFALSE;

  for (Int_t iPhE = 0; iPhE < nPhE; ++iPhE) {
    Float_t tPhE = timeDiff + mRndSignalShape.getNextValue();
    Int_t const firstBin = roundVc(TMath::Max((Int_t)0, (Int_t)((tPhE - mPMTransitTime) / mBinSize)));
    Int_t const lastBin = TMath::Min((Int_t)NBC2Cache * mNTimeBinsPerBC - 1, (Int_t)((tPhE + 2.0 * mPMTransitTime) / mBinSize));
    //LOG(INFO) << "firstBin = "<<firstBin<<" lastbin "<<lastBin;

    Float_t const tempT = mBinSize * (0.5f + firstBin) - tPhE;
    long iStart = std::lround((tempT + 2.0f * mPMTransitTime) / mBinSize);
    float const offset = tempT + 2.0f * mPMTransitTime - Float_t(iStart) * mBinSize;
    long const iOffset = std::lround(offset / mBinSize * Float_t(parameters.mNResponseTables - 1));
    if (iStart < 0) { // this should not happen
      LOG(ERROR) << "FDDDigitizer: table lookup failure";
    }
    iStart = roundVc(std::max(long(0), iStart));

    Vc::float_v workVc;
    Vc::float_v pmtVc;
    Float_t const* q = mPMResponseTables[parameters.mNResponseTables / 2 + iOffset].data() + iStart;
    Float_t const* qEnd = &mPMResponseTables[parameters.mNResponseTables / 2 + iOffset].back();

    for (int ir = firstBin / mNTimeBinsPerBC; ir <= lastBin / mNTimeBinsPerBC; ir++) {
      int localFirst = (ir == firstBin / mNTimeBinsPerBC) ? firstBin : 0;
      int localLast = (ir < lastBin / mNTimeBinsPerBC) ? mNTimeBinsPerBC : (lastBin - ir * mNTimeBinsPerBC);
      auto bcCache = getBCCache(cachedIR[ir]);
      auto& analogSignal = (*bcCache).pulse[channel];
      Float_t* p = analogSignal.data() + localFirst;

      for (int localBin = localFirst, iEnd = roundVc(localLast); q < qEnd && localBin < iEnd; localBin += Vc::float_v::Size) {
        pmtVc.load(q);
        q += Vc::float_v::Size;
        Vc::prefetchForOneRead(q);
        workVc.load(p);
        workVc += mRndGainVar.getNextValueVc() * charge * pmtVc;
        workVc.store(p);
        p += Vc::float_v::Size;
        Vc::prefetchForOneRead(p);
      }
      added[ir] = kTRUE;
    }
  }
  for (int ir = 0; ir < nCachedIR; ir++) {
    if (added[ir]) {
      auto bcCache = getBCCache(cachedIR[ir]);
      (*bcCache).labels.emplace_back(parID, mEventID, mSrcID, channel);
    }
  }
}
//_____________________________________________________________________________
void Digitizer::flush(std::vector<o2::fdd::Digit>& digitsBC,
                      std::vector<o2::fdd::ChannelData>& digitsCh,
                      o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>& labels)
{

  // do we have something to flush? We can do this only for cached BC data which is distanced from currently processed BC by NBCReadOut
  int nCached = mCache.size();
  if (nCached < 1) {
    return;
  }
  if (mIntRecord.differenceInBC(mCache.back()) > -BCCacheMin) {
    LOG(DEBUG) << "Generating new pedestal BL fluct. for BC range " << mCache.front() << " : " << mCache.back();
    //generatePedestal();
  } else {
    return;
  }
  //o2::InteractionRecord ir0(mCache.front());
  //int cacheSpan = 1 + mCache.back().differenceInBC(ir0);
  //LOG(INFO) << "Cache spans " << cacheSpan << " with " << nCached << " BCs cached";

  for (int ibc = 0; ibc < nCached; ibc++) { // digitize BCs which might not be affected by future events
    auto& bc = mCache[ibc];
    storeBC(bc, digitsBC, digitsCh, labels);
  }
  // clean cache for BCs which are not needed anymore
  //LOG(INFO) << "Cleaning cache";
  mCache.erase(mCache.begin(), mCache.end());
}
//_____________________________________________________________________________
void Digitizer::storeBC(const BCCache& bc,
                        std::vector<o2::fdd::Digit>& digitsBC, std::vector<o2::fdd::ChannelData>& digitsCh,
                        o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>& labels)
{
  //LOG(INFO) << "Storing BC " << bc;

  int first = digitsCh.size();
  for (int ic = 0; ic < mNchannels; ic++) {
    digitsCh.emplace_back(ic, SimulateTimeCFD(bc.pulse[ic]), IntegrateCharge(bc.pulse[ic]), 0, 0, 0, 0, 0, 0, 0, 0, 0);
  }
  //bc.print();

  int nBC = digitsBC.size();
  digitsBC.emplace_back(first, 16, bc, mTriggers);

  for (const auto& lbl : bc.labels)
    labels.addElement(nBC, lbl);
}

//_____________________________________________________________________________
Float_t Digitizer::IntegrateCharge(ChannelBCDataF pulse)
{
  Float_t chargeADC = 0;
  for (Int_t iBin = 0; iBin < mNTimeBinsPerBC; ++iBin) {
    //pulse[iBin] /= mChargePerADC;
    chargeADC += pulse[iBin];
  }
  //saturation if(chargeADC > )chargeADC = ;

  //LOG(INFO) <<" Charge " << chargeADC;
  return std::lround(chargeADC);
}
//_____________________________________________________________________________
Float_t Digitizer::SimulateTimeCFD(ChannelBCDataF pulse)
{

  std::fill(mTimeCFD.begin(), mTimeCFD.end(), 0);
  Float_t timeCFD = -1024;
  Int_t binShift = TMath::Nint(parameters.mTimeShiftCFD / mBinSize);
  for (Int_t iBin = 0; iBin < mNTimeBinsPerBC; ++iBin) {
    //if (mTime[channel][iBin] != 0) std::cout << mTime[channel][iBin] / parameters.mChargePerADC << ", ";
    if (iBin >= binShift)
      mTimeCFD[iBin] = 5.0 * pulse[iBin - binShift] - pulse[iBin];
    else
      mTimeCFD[iBin] = -1.0 * pulse[iBin];
  }
  for (Int_t iBin = 1; iBin < mNTimeBinsPerBC; ++iBin) {
    if (mTimeCFD[iBin - 1] < 0 && mTimeCFD[iBin] >= 0) {
      timeCFD = mBinSize * Float_t(iBin);
      break;
    }
  }
  //LOG(INFO) <<" Time " << timeCFD;
  return timeCFD;
}
//_____________________________________________________________________________
o2::fdd::Digitizer::BCCache& Digitizer::setBCCache(const o2::InteractionRecord& ir)
{
  if (mCache.empty() || mCache.back() < ir) {
    mCache.emplace_back();
    auto& cb = mCache.back();
    cb = ir;
    return cb;
  }
  if (mCache.front() > ir) {
    mCache.emplace_front();
    auto& cb = mCache.front();
    cb = ir;
    return cb;
  }

  for (auto cb = mCache.begin(); cb != mCache.end(); cb++) {
    if ((*cb) == ir) {
      return *cb;
    }
    if (ir < (*cb)) {
      auto cbnew = mCache.emplace(cb); // insert new element before cb
      (*cbnew) = ir;
      return (*cbnew);
    }
  }
  return mCache.front();
}
//_____________________________________________________________________________
o2::fdd::Digitizer::BCCache* Digitizer::getBCCache(const o2::InteractionRecord& ir)
{
  // get pointer on existing cache
  for (auto cb = mCache.begin(); cb != mCache.end(); cb++) {
    if ((*cb) == ir) {
      return &(*cb);
    }
  }
  return nullptr;
}
//_____________________________________________________________________________
void Digitizer::SetTriggers(o2::fdd::Digit* digit)
{
  //mTriggers.set
}
//_______________________________________________________________________
void Digitizer::init()
{
  auto const roundVc = [&](int i) -> int {
    return (i / Vc::float_v::Size) * Vc::float_v::Size;
  };
  // set up PMT response tables
  Float_t offset = -0.5f * mBinSize; // offset \in [-0.5..0.5] * mBinSize
  Int_t const nBins = roundVc(std::lround(4.0f * mPMTransitTime / mBinSize));
  for (auto& table : mPMResponseTables) {
    table.resize(nBins);
    Float_t t = -2.0f * mPMTransitTime + offset; // t \in offset + [-2 2] * DP::mPmtTransitTime
    for (Int_t j = 0; j < nBins; ++j) {
      table[j] = Digitizer::PMResponse(t);
      t += mBinSize;
    }
    offset += mBinSize / Float_t(parameters.mNResponseTables - 1);
  }

  TF1 scintDelayFn("fScintDelay", "gaus", -6.0f * mIntTimeRes, +6.0f * mIntTimeRes);
  scintDelayFn.SetParameters(1, 0, mIntTimeRes);
  mRndScintDelay.initialize(scintDelayFn);

  // Initialize function describing the PMT time response
  TF1 pmtResponseFn("mPmtResponseFn", &Digitizer::PMResponse, -1.0f * mPMTransitTime, +2.0f * mPMTransitTime, 0);
  pmtResponseFn.SetNpx(100);
  mPmtTimeIntegral = pmtResponseFn.Integral(-1.0f * mPMTransitTime, +2.0f * mPMTransitTime);

  // Initialize function describing PMT response to the single photoelectron
  TF1 singlePhESpectrumFn("mSinglePhESpectrum",
                          &Digitizer::SinglePhESpectrum, 0, 30, 0);
  Float_t const meansPhE = singlePhESpectrumFn.Mean(0, 30);
  mRndGainVar.initialize([&]() -> float {
    return singlePhESpectrumFn.GetRandom(0, 30) / meansPhE;
  });

  TF1 signalShapeFn("signalShape", "crystalball", 0, 300);
  signalShapeFn.SetParameters(1, parameters.mShapeSigma, parameters.mShapeSigma, parameters.mShapeAlpha, parameters.mShapeN);
  mRndSignalShape.initialize([&]() -> float {
    return signalShapeFn.GetRandom(0, 200);
  });
}
//_______________________________________________________________________
void Digitizer::finish() {}

//_____________________________________________________________________________
Int_t Digitizer::SimulateLightYield(Int_t pmt, Int_t nPhot)
{
  const Float_t p = parameters.mLightYield * mPhotoCathodeEfficiency;
  if (p == 1.0f || nPhot == 0)
    return nPhot;
  const Int_t n = Int_t(nPhot < 100 ? gRandom->Binomial(nPhot, p) : gRandom->Gaus(p * nPhot + 0.5, TMath::Sqrt(p * (1 - p) * nPhot)));
  return n;
}
//_____________________________________________________________________________
Double_t Digitizer::PMResponse(Double_t* x, Double_t*)
{
  return Digitizer::PMResponse(x[0]);
}
//_____________________________________________________________________________
Double_t Digitizer::PMResponse(Double_t x)
{
  // this function describes the PM time response to a single photoelectron
  Double_t y = x + mPMTransitTime;
  return y * y * TMath::Exp(-y * y / (mPMTransitTime * mPMTransitTime));
}
//_____________________________________________________________________________
Double_t Digitizer::SinglePhESpectrum(Double_t* x, Double_t*)
{
  // this function describes the PM amplitude response to a single photoelectron
  Double_t y = x[0];
  if (y < 0)
    return 0;
  return (TMath::Poisson(y, mPMNbOfSecElec) + mPMTransparency * TMath::Poisson(y, 1.0));
}
//______________________________________________________________
void Digitizer::BCCache::print() const
{
  printf("Cached Orbit:%5d/BC:%4d", orbit, bc);
  for (int ic = 0; ic < 16; ic++) {
    printf("Ch[%d] | ", ic);
    for (int ib = 0; ib < mNTimeBinsPerBC; ib++) {
      if (ib % 10 == 0)
        printf("%f ", pulse[ic][ib]);
    }
    printf("\n");
  }
}
