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

//_____________________________________________________________________________
void Digitizer::process(const std::vector<o2::fdd::Hit>* hits, o2::fdd::Digit* digit)
{
  auto sorted_hits{*hits};
  std::sort(sorted_hits.begin(), sorted_hits.end(), [](o2::fdd::Hit const& a, o2::fdd::Hit const& b) {
    return a.GetTrackID() < b.GetTrackID();
  });
  digit->SetTime(mEventTime);
  digit->SetInteractionRecord(mIntRecord);

  std::vector<o2::fdd::ChannelData>& channel_data = digit->GetChannelData();
  if (channel_data.size() == 0) {
    channel_data.reserve(parameters.mNchannels);
    for (int i = 0; i < parameters.mNchannels; ++i)
      channel_data.emplace_back(o2::fdd::ChannelData{i, o2::InteractionRecord::DummyTime, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  }

  auto const roundVc = [&](int i) -> int {
    return (i / Vc::float_v::Size) * Vc::float_v::Size;
  };
  Int_t parent = -10;
  for (Int_t i = 0; i < parameters.mNchannels; i++)
    std::fill(mTime[i].begin(), mTime[i].end(), 0);

  assert(digit->GetChannelData().size() == parameters.mNchannels);
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
    Int_t pmt = hit.GetDetectorID();
    Int_t nPhE = SimulateLightYield(pmt, hit.GetNphot());

    Float_t dt_scintillator = mRndScintDelay.getNextValue();
    Float_t t = dt_scintillator + hit.GetTime();
    // LOG(INFO) << "Nphot = " << hit.GetNphot() << " time =" << hit.GetTime();
    // LOG(INFO) << "NphE = " << nPhE;
    Float_t charge = TMath::Qe() * parameters.mPmGain * mBinSize / mPmtTimeIntegral;

    auto& analogSignal = mTime[pmt];

    for (Int_t iPhE = 0; iPhE < nPhE; ++iPhE) {
      Float_t tPhE = t + mRndSignalShape.getNextValue();
      //LOG(INFO) <<"t = "<<t<<"tPhE = "<<tPhE;
      Int_t const firstBin = roundVc(TMath::Max((Int_t)0, (Int_t)((tPhE - parameters.mPMTransitTime) / mBinSize)));
      Int_t const lastBin = TMath::Min((Int_t)mNBins - 1, (Int_t)((tPhE + 2. * parameters.mPMTransitTime) / mBinSize));
      //LOG(INFO) << "firstBin = "<<firstBin<<" lastbin "<<lastBin;
      Float_t const tempT = mBinSize * (0.5f + firstBin) - tPhE;
      Float_t* p = analogSignal.data() + firstBin;
      long iStart = std::lround((tempT + 2.0f * parameters.mPMTransitTime) / mBinSize);
      float const offset = tempT + 2.0f * parameters.mPMTransitTime - Float_t(iStart) * mBinSize;
      long const iOffset = std::lround(offset / mBinSize * Float_t(parameters.mNResponseTables - 1));
      if (iStart < 0) { // this should not happen
        LOG(ERROR) << "FDDDigitizer: table lookup failure";
      }
      iStart = roundVc(std::max(long(0), iStart));

      Vc::float_v workVc;
      Vc::float_v pmtVc;
      Float_t const* q = mPMResponseTables[parameters.mNResponseTables / 2 + iOffset].data() + iStart;
      Float_t const* qEnd = &mPMResponseTables[parameters.mNResponseTables / 2 + iOffset].back();
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
    }
    //MCLabels
    Int_t parentID = hit.GetTrackID();
    if (parentID != parent) {
      o2::fdd::MCLabel label(hit.GetTrackID(), mEventID, mSrcID, pmt);
      if (mMCLabels)
        mMCLabels->addElement(mMCLabels->getIndexedSize(), label);
      parent = parentID;
    } //labels
  }   //hit loop

  //Conversion of analogue pulse shape to values provided by FEE
  for (Int_t ipmt = 0; ipmt < parameters.mNchannels; ++ipmt) {
    channel_data[ipmt].mTime = SimulateTimeCFD(ipmt);
    for (Int_t iBin = 0; iBin < mNBins; ++iBin)
      channel_data[ipmt].mChargeADC += mTime[ipmt][iBin] / parameters.mChargePerADC;
    // LOG(INFO) << "ADC " << channel_data[ipmt].mChargeADC << " Time " << channel_data[ipmt].mTime;
  }
}
//_____________________________________________________________________________
Float_t Digitizer::SimulateTimeCFD(Int_t channel)
{

  std::fill(mTimeCFD.begin(), mTimeCFD.end(), 0);
  Float_t timeCFD = -1024;
  Int_t binShift = TMath::Nint(parameters.mTimeShiftCFD / mBinSize);
  for (Int_t iBin = 0; iBin < mNBins; ++iBin) {
    //if (mTime[channel][iBin] != 0) std::cout << mTime[channel][iBin] / parameters.mChargePerADC << ", ";
    if (iBin >= binShift)
      mTimeCFD[iBin] = 5.0 * mTime[channel][iBin - binShift] - mTime[channel][iBin];
    else
      mTimeCFD[iBin] = -1.0 * mTime[channel][iBin];
  }
  for (Int_t iBin = 1; iBin < mNBins; ++iBin) {
    if (mTimeCFD[iBin - 1] < 0 && mTimeCFD[iBin] >= 0) {
      timeCFD = mBinSize * Float_t(iBin);
      break;
    }
  }

  return timeCFD;
}
//_____________________________________________________________________________
void Digitizer::SetTriggers(o2::fdd::Digit* digit)
{
}
//_____________________________________________________________________________
void Digitizer::initParameters()
{
  mEventTime = 0;
}
//_______________________________________________________________________
void Digitizer::init()
{
  mEventTime = 0;

  mNBins = 2000;           //Will be computed using detector set-up from CDB
  mBinSize = 25.0 / 256.0; //Will be set-up from CDB
  for (Int_t i = 0; i < parameters.mNchannels; i++)
    mTime[i].resize(mNBins);
  mTimeCFD.resize(mNBins);

  auto const roundVc = [&](int i) -> int {
    return (i / Vc::float_v::Size) * Vc::float_v::Size;
  };
  // set up PMT response tables
  Float_t offset = -0.5f * mBinSize; // offset \in [-0.5..0.5] * mBinSize
  Int_t const nBins = roundVc(std::lround(4.0f * parameters.mPMTransitTime / mBinSize));
  for (auto& table : mPMResponseTables) {
    table.resize(nBins);
    Float_t t = -2.0f * parameters.mPMTransitTime + offset; // t \in offset + [-2 2] * DP::mPmtTransitTime
    for (Int_t j = 0; j < nBins; ++j) {
      table[j] = Digitizer::PMResponse(t);
      t += mBinSize;
    }
    offset += mBinSize / Float_t(parameters.mNResponseTables - 1);
  }

  TF1 scintDelayFn("fScintDelay", "gaus", -6.0f * parameters.mIntTimeRes, +6.0f * parameters.mIntTimeRes);
  scintDelayFn.SetParameters(1, 0, parameters.mIntTimeRes);
  mRndScintDelay.initialize(scintDelayFn);

  // Initialize function describing the PMT time response
  TF1 pmtResponseFn("mPmtResponseFn", &Digitizer::PMResponse, -1.0f * parameters.mPMTransitTime, +2.0f * parameters.mPMTransitTime, 0);
  pmtResponseFn.SetNpx(100);
  mPmtTimeIntegral = pmtResponseFn.Integral(-1.0f * parameters.mPMTransitTime, +2.0f * parameters.mPMTransitTime);

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
    return signalShapeFn.GetRandom(0, mBinSize * Float_t(mNBins));
  });
}
//_______________________________________________________________________
void Digitizer::finish() {}

//_____________________________________________________________________________
Int_t Digitizer::SimulateLightYield(Int_t pmt, Int_t nPhot)
{
  const Float_t p = parameters.mLightYield * parameters.mPhotoCathodeEfficiency;
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
  Double_t y = x + DigitizationParameters::mPMTransitTime;
  return y * y * TMath::Exp(-y * y / (DigitizationParameters::mPMTransitTime * DigitizationParameters::mPMTransitTime));
}
//_____________________________________________________________________________
Double_t Digitizer::SinglePhESpectrum(Double_t* x, Double_t*)
{
  // this function describes the PM amplitude response to a single photoelectron
  Double_t y = x[0];
  if (y < 0)
    return 0;
  return (TMath::Poisson(y, DigitizationParameters::mPMNbOfSecElec) + DigitizationParameters::mPMTransparency * TMath::Poisson(y, 1.0));
}
