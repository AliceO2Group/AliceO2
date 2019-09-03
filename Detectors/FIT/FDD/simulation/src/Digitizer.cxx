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
#include "MathUtils/CachingTF1.h"
#include <algorithm>
#include <cassert>
#include <iostream>

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
  Int_t parent = -10;
  Float_t integral = mPMResponse->Integral(-parameters.mPMTransitTime, 2. * parameters.mPMTransitTime);
  Float_t meansPhE = mSinglePhESpectrum->Mean(0, 20);
  for (Int_t i = 0; i < parameters.mNchannels; i++)
    std::fill(mTime[i].begin(), mTime[i].end(), 0);

  assert(digit->GetChannelData().size() == parameters.mNchannels);
  //Conversion of hits to the analogue pulse shape
  for (auto& hit : sorted_hits) {
    Int_t pmt = hit.GetDetectorID();
    Int_t nPhE = SimulateLightYield(pmt, hit.GetNphot());

    Float_t dt_scintillator = gRandom->Gaus(0, parameters.mIntTimeRes);
    Float_t t = dt_scintillator + hit.GetTime();

    //LOG(INFO) << "Nphot = "<<hit.GetNphot()<<FairLogger::endl;
    //LOG(INFO) << "NphE = " << nPhE << FairLogger::endl;
    Float_t charge = TMath::Qe() * parameters.mPmGain * mBinSize / integral;
    for (Int_t iPhE = 0; iPhE < nPhE; ++iPhE) {
      Float_t tPhE = t + mSignalShape->GetRandom(0, mBinSize * Float_t(mNBins));
      Float_t gainVar = mSinglePhESpectrum->GetRandom(0, 20) / meansPhE;
      Int_t firstBin = TMath::Max((UInt_t)0, (UInt_t)((tPhE - parameters.mPMTransitTime) / mBinSize));
      Int_t lastBin = TMath::Min(mNBins - 1, (UInt_t)((tPhE + 2. * parameters.mPMTransitTime) / mBinSize));
      //LOG(INFO) << "firstBin = "<<firstBin<<" lastbin "<<lastBin<<FairLogger::endl;
      for (Int_t iBin = firstBin; iBin <= lastBin; ++iBin) {
        Float_t tempT = mBinSize * (0.5 + iBin) - tPhE;
        mTime[pmt][iBin] += gainVar * charge * mPMResponse->Eval(tempT);
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
    //LOG(INFO) << "ADC " << channel_data[ipmt].mChargeADC << " Time " << channel_data[ipmt].mTime << FairLogger::endl;
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
  // this function describes the PM time response to a single photoelectron
  Double_t y = x[0] + parameters.mPMTransitTime;
  return y * y * TMath::Exp(-y * y / (parameters.mPMTransitTime * parameters.mPMTransitTime));
}
//_____________________________________________________________________________
Double_t Digitizer::SinglePhESpectrum(Double_t* x, Double_t*)
{
  // this function describes the PM amplitude response to a single photoelectron
  Double_t y = x[0];
  if (y < 0)
    return 0;
  return (TMath::Poisson(y, parameters.mPMNbOfSecElec) + parameters.mPMTransparency * TMath::Poisson(y, 1.0));
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

  mNBins = 2000;           //Will be computed using detector set-up from CDB
  mBinSize = 25.0 / 256.0; //Will be set-up from CDB
  for (Int_t i = 0; i < parameters.mNchannels; i++)
    mTime[i].resize(mNBins);
  mTimeCFD.resize(mNBins);

  if (!mPMResponse)
    mPMResponse = std::make_unique<o2::base::CachingTF1>("mPMResponse", this, &Digitizer::PMResponse, -parameters.mPMTransitTime, 2. * parameters.mPMTransitTime, 0);
  if (!mSinglePhESpectrum)
    mSinglePhESpectrum = std::make_unique<o2::base::CachingTF1>("mSinglePhESpectrum", this, &Digitizer::SinglePhESpectrum, 0, 20, 0);
  if (!mSignalShape) {
    mSignalShape = std::make_unique<o2::base::CachingTF1>("mSignalShape", "crystalball", 0, 300);
    mSignalShape->SetParameters(1, parameters.mShapeSigma, parameters.mShapeSigma, parameters.mShapeAlpha, parameters.mShapeN);
  }
}
//_______________________________________________________________________
void Digitizer::finish() {}
