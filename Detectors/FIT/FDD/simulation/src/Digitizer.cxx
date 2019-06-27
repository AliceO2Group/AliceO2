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

using namespace o2::fdd;

ClassImp(Digitizer);

//_____________________________________________________________________________
void Digitizer::process(const std::vector<o2::fdd::Hit>* hits, o2::fdd::Digit* digit)
{
  auto sorted_hits{ *hits };
  std::sort(sorted_hits.begin(), sorted_hits.end(), [](o2::fdd::Hit const& a, o2::fdd::Hit const& b) {
    return a.GetTrackID() < b.GetTrackID();
  });
  digit->SetTime(mEventTime);
  digit->SetInteractionRecord(mIntRecord);

  std::vector<UInt_t> nPMHits;
  nPMHits.reserve(parameters.mNchannels);
  for (int i = 0; i < parameters.mNchannels; ++i)
    nPMHits[i] = 0; 
  std::vector<o2::fdd::ChannelData>& channel_data = digit->GetChannelData();
  if (channel_data.size() == 0) {
    channel_data.reserve(parameters.mNchannels);
    for (int i = 0; i < parameters.mNchannels; ++i)
      channel_data.emplace_back(o2::fdd::ChannelData{ i, 1024, 0, 0 });
  }
  Int_t parent = -10;
  assert(digit->GetChannelData().size() == parameters.mNchannels);
  for (auto& hit : sorted_hits) {
    const Int_t pmt = hit.GetDetectorID();
    const Int_t nPhot = SimulateLightYield(pmt, hit.GetNphot());
    const Float_t dt_scintillator = gRandom->Gaus(0,parameters.mIntTimeRes);
    const Float_t t = dt_scintillator + hit.GetTime();
    channel_data[pmt].mChargeADC += nPhot;
    nPMHits[pmt]++;
    if (channel_data[pmt].mTime>t) channel_data[pmt].mTime = t;
    
    //MCLabels
    Int_t parentID = hit.GetTrackID();
    if (parentID != parent) {
      o2::fdd::MCLabel label(hit.GetTrackID(), mEventID, mSrcID, pmt);
      int lblCurrent;
      if (mMCLabels) {
        lblCurrent = mMCLabels->getIndexedSize(); // this is the size of mHeaderArray;
        mMCLabels->addElement(lblCurrent, label);
      }
      parent = parentID;
    } 
  }
  for (int i = 0; i < parameters.mNchannels; ++i)if (nPMHits[i]==0 || channel_data[i].mChargeADC==0) channel_data[i].mTime = 0.0; 
}

//_____________________________________________________________________________
Int_t Digitizer::SimulateLightYield(Int_t pmt, Int_t nPhot)
{
  const Float_t p = 1.0;
  if (p == 1.0f || nPhot == 0)
    return nPhot;
  const Int_t n = Int_t(0.5+1.0f/p*(nPhot < 100 ? gRandom->Binomial(nPhot, p) : gRandom->Gaus(p*nPhot+0.5, TMath::Sqrt(p*(1-p)*nPhot))));
  return n;
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
void Digitizer::init(){}
//_______________________________________________________________________
void Digitizer::finish(){}

