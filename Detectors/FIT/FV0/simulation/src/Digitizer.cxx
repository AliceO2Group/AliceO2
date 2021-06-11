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
#include "FV0Base/Geometry.h"
#include "FV0Base/Constants.h"

#include <TRandom.h>
#include <algorithm>
#include <numeric>

ClassImp(o2::fv0::Digitizer);

using namespace o2::math_utils;
using namespace o2::fv0;

void Digitizer::clear()
{
  mEventId = -1;
  mSrcId = -1;
  for (auto& analogSignal : mPmtChargeVsTime) {
    std::fill_n(std::begin(analogSignal), analogSignal.size(), 0);
  }
  mLastBCCache.clear();
  mCfdStartIndex.fill(0);
}

//_______________________________________________________________________
void Digitizer::init()
{
  LOG(INFO) << "V0Digitizer::init -> start = ";
  mNBins = FV0DigParam::Instance().waveformNbins;      //Will be computed using detector set-up from CDB
  mBinSize = FV0DigParam::Instance().waveformBinWidth; //Will be set-up from CDB

  mNTimeBinsPerBC = std::lround(o2::constants::lhc::LHCBunchSpacingNS / mBinSize); // 1920 bins/BC

  for (Int_t detID = 0; detID < Constants::nFv0Channels; detID++) {
    mPmtChargeVsTime[detID].resize(mNBins);
    mLastBCCache.mPmtChargeVsTime[detID].resize(mNBins);
  }

  // set up PMT response function [avg]
  TF1 signalShapeFn("signalShape", "crystalball", 0, 200);
  signalShapeFn.SetParameters(FV0DigParam::Instance().shapeConst,
                              FV0DigParam::Instance().shapeMean,
                              FV0DigParam::Instance().shapeSigma,
                              FV0DigParam::Instance().shapeAlpha,
                              FV0DigParam::Instance().shapeN);

  // PMT response per hit [Global]
  double x = mBinSize / 2.0; /// Calculate at BinCenter
  mPmtResponseGlobal.resize(mNBins);
  for (auto& y : mPmtResponseGlobal) {
    y = signalShapeFn.Eval(x);
    //LOG(INFO)<<x<<"    "<<mPmtResponseGlobal[j];
    x += mBinSize;
  }

  mLastBCCache.clear();
  mCfdStartIndex.fill(0);

  LOG(INFO) << "V0Digitizer::init -> finished";
}

void Digitizer::process(const std::vector<o2::fv0::Hit>& hits,
                        std::vector<o2::fv0::BCData>& digitsBC,
                        std::vector<o2::fv0::ChannelData>& digitsCh,
                        std::vector<o2::fv0::DetTrigInput>& digitsTrig,
                        o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>& labels)
{
  LOG(INFO) << "[FV0] Digitizer::process(): begin with " << hits.size() << " hits";
  flush(digitsBC, digitsCh, digitsTrig, labels); // flush cached signal which cannot be affect by new event

  std::vector<int> hitIdx(hits.size());
  std::iota(std::begin(hitIdx), std::end(hitIdx), 0);
  std::sort(std::begin(hitIdx), std::end(hitIdx),
            [&hits](int a, int b) { return hits[a].GetTrackID() < hits[b].GetTrackID(); });

  // use ordered hits
  for (auto ids : hitIdx) {
    const auto& hit = hits[ids];
    Int_t detId = hit.GetDetectorID();
    Double_t hitEdep = hit.GetHitValue() * 1e3; //convert to MeV
    Float_t const hitTime = hit.GetTime() * 1e9;
    // TODO: check how big is inaccuracy if more than 1 'below-threshold' particles hit the same detector cell
    if (hitEdep < FV0DigParam::Instance().singleMipThreshold || hitTime > FV0DigParam::Instance().singleHitTimeThreshold) {
      continue;
    }
    float distanceFromXc = 0;
    if (Geometry::instance()->isRing5(detId)) {
      distanceFromXc = getDistFromCellCenter(detId, hit.GetX(), hit.GetY());
    }

    int iChannelPerCell = 0;
    while (iChannelPerCell < 2) { // loop over 2 channels, into which signal from each cell in ring 5 is split
      if (Geometry::instance()->isRing5(detId)) {
        // The first channel number is located counter-clockwise from the cell center
        // and remains identical to the detector number, the second one is clockwise and incremented by 8
        if (iChannelPerCell == 1) {
          detId += 8;
        }
        // Split signal magnitude to fractions depending on the distance of the hit from the cell center
        hitEdep = (hit.GetHitValue() * 1e3) * getSignalFraction(distanceFromXc, iChannelPerCell == 0);
        // LOG(INFO) << "  detId: " << detId << "-" << iChannelPerCell << " hitEdep: " << hitEdep << " distanceFromXc: " << distanceFromXc;
        ++iChannelPerCell;
      } else {
        iChannelPerCell = 2; // not a ring 5 cell -> don't repeat the loop
      }
      Double_t const nPhotons = hitEdep * DP::N_PHOTONS_PER_MEV;
      float const nPhE = SimulateLightYield(detId, nPhotons);
      float const mipFraction = float(nPhE / FV0DigParam::Instance().avgNumberPhElectronPerMip);
      Float_t timeHit = hitTime;
      timeHit += mIntRecord.getTimeNS();
      o2::InteractionTimeRecord const irHit(timeHit);
      std::array<o2::InteractionRecord, NBC2Cache> cachedIR;
      int nCachedIR = 0;
      for (int i = BCCacheMin; i < BCCacheMax + 1; i++) {
        double const tNS = timeHit + o2::constants::lhc::LHCBunchSpacingNS * i;
        cachedIR[nCachedIR].setFromNS(tNS);
        if (tNS < 0 && cachedIR[nCachedIR] > irHit) {
          continue; // don't go to negative BC/orbit (it will wrap)
        }
        setBCCache(cachedIR[nCachedIR++]); // ensure existence of cached container
      }                                    //BCCache loop
      createPulse(mipFraction, hit.GetTrackID(), hitTime, cachedIR, nCachedIR, detId);

    } //while loop
  }   //hitloop
}

void Digitizer::createPulse(float mipFraction, int parID, const double hitTime,
                            std::array<o2::InteractionRecord, NBC2Cache> const& cachedIR, int nCachedIR, const int detId)
{

  std::array<bool, NBC2Cache> added;
  added.fill(false);

  for (int ir = 0; ir < NBC2Cache; ir++) {
    auto bcCache = getBCCache(cachedIR[ir]);
    for (int ich = 0; ich < Constants::nFv0Channels; ich++) {
      (*bcCache).mPmtChargeVsTime[ich].resize(mNTimeBinsPerBC);
    }
  }

  ///Time of flight subtracted from Hit time //TODO have different TOF according to thr ring number
  Int_t const NBinShift = std::lround((hitTime - FV0DigParam::Instance().globalTimeOfFlight) / FV0DigParam::Instance().waveformBinWidth);

  if (NBinShift >= 0 && NBinShift < FV0DigParam::Instance().waveformNbins) {
    mPmtResponseTemp.resize(FV0DigParam::Instance().waveformNbins, 0.);
    std::memcpy(&mPmtResponseTemp[NBinShift], &mPmtResponseGlobal[0],
                sizeof(double) * (FV0DigParam::Instance().waveformNbins - NBinShift));
  } else {
    mPmtResponseTemp = mPmtResponseGlobal;
    mPmtResponseTemp.erase(mPmtResponseTemp.begin(), mPmtResponseTemp.begin() + abs(NBinShift));
    mPmtResponseTemp.resize(FV0DigParam::Instance().waveformNbins);
  }

  for (int ir = 0; ir < int(mPmtResponseTemp.size() / mNTimeBinsPerBC); ir++) {
    auto bcCache = getBCCache(cachedIR[ir]);

    for (int iBin = 0; iBin < mNTimeBinsPerBC; iBin++) {
      (*bcCache).mPmtChargeVsTime[detId][iBin] += (mPmtResponseTemp[ir * mNTimeBinsPerBC + iBin] * mipFraction);
    }
    added[ir] = true;
  }
  ///Add MC labels to BCs for those contributed to the PMT signal
  for (int ir = 0; ir < nCachedIR; ir++) {
    if (added[ir]) {
      auto bcCache = getBCCache(cachedIR[ir]);
      (*bcCache).labels.emplace_back(parID, mEventId, mSrcId, detId);
    }
  }
}

void Digitizer::flush(std::vector<o2::fv0::BCData>& digitsBC,
                      std::vector<o2::fv0::ChannelData>& digitsCh,
                      std::vector<o2::fv0::DetTrigInput>& digitsTrig,
                      o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>& labels)
{
  for (auto const& bc : mCache) {
    if (mIntRecord.differenceInBC(bc) > NBC2Cache) { // Build events that are separated by NBC2Cache BCs from current BC
      storeBC(bc, digitsBC, digitsCh, digitsTrig, labels);
      mCache.pop_front();
    } else {
      return;
    }
  }
}

void Digitizer::storeBC(const BCCache& bc,
                        std::vector<o2::fv0::BCData>& digitsBC,
                        std::vector<o2::fv0::ChannelData>& digitsCh,
                        std::vector<o2::fv0::DetTrigInput>& digitsTrig,
                        o2::dataformats::MCTruthContainer<o2::fv0::MCLabel>& labels)

{
  size_t const nBC = digitsBC.size();   // save before digitsBC is being modified
  size_t const first = digitsCh.size(); // save before digitsCh is being modified
  size_t nStored = 0;
  double totalChargeAllRing = 0;
  double nSignalInner = 0;
  double nSignalOuter = 0;

  if (mLastBCCache.differenceInBC(bc) != 1) { // if the last buffered BC is not the one before the current BC
    mLastBCCache.clear();                     // clear the bufffer (mPmtChargeVsTime set to 0s)
    mCfdStartIndex.fill(0);                   // reset all start indices to 0, i.e., to the beginning of the BC
  }

  for (int iPmt = 0; iPmt < Constants::nFv0Channels; iPmt++) {
    // run the CFD: this updates the start index for the next BC in case the CFD dead time ends in the next BC
    double cfdWithOffset = SimulateTimeCfd(mCfdStartIndex[iPmt], mLastBCCache.mPmtChargeVsTime[iPmt], bc.mPmtChargeVsTime[iPmt]);
    double cfdZero = cfdWithOffset - FV0DigParam::Instance().avgCfdTimeForMip;

    if (cfdZero < -FV0DigParam::Instance().cfdCheckWindow || cfdZero > FV0DigParam::Instance().cfdCheckWindow) {
      continue;
    }
    float totalCharge = IntegrateCharge(bc.mPmtChargeVsTime[iPmt]);
    totalChargeAllRing += totalCharge;
    totalCharge *= DP::INV_CHARGE_PER_ADC;
    cfdZero *= DP::INV_TIME_PER_TDCCHANNEL;

    digitsCh.emplace_back(iPmt, std::lround(cfdZero), std::lround(totalCharge));
    ++nStored;
    //---trigger---
    if (iPmt < 25) {
      nSignalInner++;
    } else {
      nSignalOuter++;
    }
  }
  // save BC information for the CFD detector
  mLastBCCache = bc;
  if (nStored < 1) {
    return;
  }
  totalChargeAllRing *= DP::INV_CHARGE_PER_ADC;
  //LOG(INFO)<<"Total charge ADC " <<totalChargeAllRing ;
  ///Triggers for FV0
  bool isMinBias, isMinBiasInner, isMinBiasOuter, isHighMult, isDummy;
  isMinBias = nStored > 0;
  isMinBiasInner = nSignalInner > 0; //ring 1,2 and 3
  isMinBiasOuter = nSignalOuter > 0; //ring 4 and 5
  isHighMult = totalChargeAllRing > FV0DigParam::Instance().adcChargeHighMultTh;
  isDummy = false;

  Triggers triggers;
  triggers.setTriggers(isMinBias, isMinBiasInner, isMinBiasOuter, isHighMult, isDummy, nStored, totalChargeAllRing);
  digitsBC.emplace_back(first, nStored, bc, triggers);
  digitsTrig.emplace_back(bc, isMinBias, isMinBiasInner, isMinBiasOuter, isHighMult, isDummy);
  for (auto const& lbl : bc.labels) {
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
  const Float_t p = FV0DigParam::Instance().lightYield * FV0DigParam::Instance().photoCathodeEfficiency;
  if ((fabs(1.0f - p) < epsilon) || nPhot == 0) {
    return nPhot;
  }
  const Int_t n = Int_t(nPhot < 100
                          ? gRandom->Binomial(nPhot, p)
                          : gRandom->Gaus((p * nPhot) + 0.5, TMath::Sqrt(p * (1. - p) * nPhot)));
  return n;
}
//---------------------------------------------------------------------------
Float_t Digitizer::IntegrateCharge(const ChannelBCDataF& pulse) const
{
  int const chargeIntMin = FV0DigParam::Instance().isIntegrateFull ? 0 : (FV0DigParam::Instance().avgCfdTimeForMip - 6.0) / mBinSize;                //Charge integration offset (cfd mean time - 6 ns)
  int const chargeIntMax = FV0DigParam::Instance().isIntegrateFull ? mNTimeBinsPerBC : (FV0DigParam::Instance().avgCfdTimeForMip + 14.0) / mBinSize; //Charge integration offset (cfd mean time + 14 ns)
  if (chargeIntMin < 0 || chargeIntMin > mNTimeBinsPerBC || chargeIntMax > mNTimeBinsPerBC) {
    LOG(FATAL) << "invalid indicess: chargeInMin=" << chargeIntMin << " chargeIntMax=" << chargeIntMax;
  }

  Float_t totalCharge = 0.0f;
  for (int iTimeBin = chargeIntMin; iTimeBin < chargeIntMax; iTimeBin++) {
    totalCharge += pulse[iTimeBin];
  }
  return totalCharge;
}
//---------------------------------------------------------------------------
Float_t Digitizer::SimulateTimeCfd(int& startIndex, const ChannelBCDataF& pulseLast, const ChannelBCDataF& pulse) const
{
  Float_t timeCfd = -1024.0f;

  if (pulse.empty()) {
    startIndex = 0;
    return timeCfd;
  }

  Float_t const cfdThrInCoulomb = FV0DigParam::Instance().mCFD_trsh * 1e-3 / 50 * mBinSize * 1e-9; // convert mV into Coulomb assuming 50 Ohm

  Int_t const binShift = TMath::Nint(FV0DigParam::Instance().timeShiftCfd / mBinSize);
  Float_t sigPrev = 5 * pulseLast[mNTimeBinsPerBC - binShift - 1] - pulseLast[mNTimeBinsPerBC - 1]; //  CFD output from the last bin of the last BC
  for (Int_t iTimeBin = 0; iTimeBin < mNTimeBinsPerBC; ++iTimeBin) {
    Float_t const sigCurrent = 5.0f * (iTimeBin >= binShift ? pulse[iTimeBin - binShift] : pulseLast[mNTimeBinsPerBC - binShift + iTimeBin]) - pulse[iTimeBin];
    if (iTimeBin >= startIndex && std::abs(pulse[iTimeBin]) > cfdThrInCoulomb) { // enable
      if (sigPrev < 0.0f && sigCurrent >= 0.0f) {                                // test for zero-crossing
        timeCfd = Float_t(iTimeBin) * mBinSize;
        startIndex = iTimeBin + std::lround(FV0DigParam::Instance().mCFDdeadTime / mBinSize); // update startIndex (CFD dead time)
        if (startIndex < mNTimeBinsPerBC) {
          startIndex = 0; // dead-time ends in same BC: no impact on the following BC
        } else {
          startIndex -= mNTimeBinsPerBC;
        }
        if (startIndex > mNTimeBinsPerBC) {
          LOG(FATAL) << "CFD dead-time was set to > 25 ns";
        }
        break; // only detects the 1st zero-crossing in the BC
      }
    }
    sigPrev = sigCurrent;
  }
  return timeCfd;
}

float Digitizer::getDistFromCellCenter(UInt_t cellId, double hitx, double hity)
{
  Geometry* geo = Geometry::instance();

  // Parametrize the line (ax+by+c=0) that crosses the detector center and the cell's middle point
  Point3Dsimple* pCell = &geo->getCellCenter(cellId);
  float x0, y0, z0;
  geo->getGlobalPosition(x0, y0, z0);
  double a = -(y0 - pCell->y) / (x0 - pCell->x);
  double b = 1;
  double c = -(y0 - a * x0);
  //Return the distance from hit to this line
  return (a * hitx + b * hity + c) / TMath::Sqrt(a * a + b * b);
}

float Digitizer::getSignalFraction(float distanceFromXc, bool isFirstChannel)
{
  float const fraction = sigmoidPmtRing5(distanceFromXc);
  if (distanceFromXc > 0) {
    return isFirstChannel ? fraction : (1. - fraction);
  } else {
    return isFirstChannel ? (1. - fraction) : fraction;
  }
}

//_____________________________________________________________________________
o2::fv0::Digitizer::BCCache& Digitizer::setBCCache(const o2::InteractionRecord& ir)
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
o2::fv0::Digitizer::BCCache* Digitizer::getBCCache(const o2::InteractionRecord& ir)
{
  // get pointer on existing cache
  for (auto cb = mCache.begin(); cb != mCache.end(); cb++) {
    if ((*cb) == ir) {
      return &(*cb);
    }
  }
  return nullptr;
}
