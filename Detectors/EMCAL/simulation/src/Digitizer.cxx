// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "SimulationDataFormat/MCCompLabel.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/GeometryBase.h"
#include "EMCALBase/Hit.h"
#include "EMCALBase/Digit.h"
#include "EMCALSimulation/Digitizer.h"
#include "MathUtils/Cartesian3D.h"

#include "FairLogger.h"   // for LOG
#include <TRandom.h>
#include <TVector3.h>
#include <climits>

ClassImp(o2::EMCAL::Digitizer);

//using o2::EMCAL::GeometryBase;
using o2::EMCAL::Hit;
using o2::EMCAL::Digit;

using namespace o2::EMCAL;

//_______________________________________________________________________
void Digitizer::init()
{
  /*
  const Int_t numOfChips = mGeometry->getNumberOfChips();

  if (mParams.getHit2DigitsMethod() == DigiParams::p2dCShape && !mParams.getAlpSimResponse()) {
    mAlpSimResp = std::make_unique<o2::ITSMFT::AlpideSimResponse>();
    mAlpSimResp->initData();
    mParams.setAlpSimResponse(mAlpSimResp.get());
  }
    
  for (Int_t i = 0; i < numOfChips; i++) {
    mSimulations.emplace_back(&mParams, i, &mGeometry->getMatrixL2G(i));
  }
  */
}

//_______________________________________________________________________
void Digitizer::process(const std::vector<Hit>* hits, std::vector<Digit>* digits)
{
  digits->clear();
  
  for(auto hit : *hits) {
    Digit digit = HitToDigit(hit);
    digits->push_back(digit);
  }

  /*
  // digitize single event
  
  const Int_t numOfChips = mGeometry->getNumberOfChips();  

  // estimate the smalles RO Frame this event may have
  double hTime0 = mEventTime - mParams.getTimeOffset();
  if (hTime0 > UINT_MAX) {
    LOG(WARNING) << "min Hit RO Frame undefined: time: " << hTime0 << " is in far future: "
		 << " EventTime: " << mEventTime << " TimeOffset: "
		 << mParams.getTimeOffset() << FairLogger::endl;
    return;
  }

  if (hTime0<0) hTime0 = 0.;
  UInt_t minNewROFrame = static_cast<UInt_t>(hTime0/mParams.getROFrameLenght());

  LOG(INFO) << "Digitizing ITS event at time " << mEventTime
	    << " (TOffset= " << mParams.getTimeOffset() << " ROFrame= " << minNewROFrame << ")"
	    << " cont.mode: " << isContinuous() << " current Min/Max RO Frames "
	    << mROFrameMin << "/" << mROFrameMax << FairLogger::endl ;
  
  if (mParams.isContinuous() && minNewROFrame>mROFrameMin) {
    // if there are already digits cached for previous RO Frames AND the new event
    // cannot contribute to these digits, move them to the output container
    if (mROFrameMax<minNewROFrame) mROFrameMax = minNewROFrame-1;
    for (auto rof=mROFrameMin; rof<minNewROFrame; rof++) {
      fillOutputContainer(digits, rof);
    }
    //    fillOutputContainer(digits, minNewROFrame-1);
  }
  
  // accumulate hits for every chip
  for(auto& hit : *hits) {
    // RS: ATTENTION: this is just a trick until we clarify how the hits from different source are
    // provided and identified. At the moment we just create a combined identifier from eventID
    // and sourceID and store it TEMPORARILY in the cached Point's TObject UniqueID
    const_cast<Hit&>(hit).SetSrcEvID(mCurrSrcID,mCurrEvID); 
    mSimulations[hit.GetDetectorID()].InsertHit(&hit);
  }
    
  // Convert hits to digits  
  for (auto &simulation : mSimulations) {
    simulation.Hits2Digits(mEventTime, mROFrameMin, mROFrameMax);
    simulation.ClearHits();
  }

  // in the triggered mode store digits after every MC event
  if (!mParams.isContinuous()) {
    fillOutputContainer(digits, mROFrameMax);
  }
  */
}

//_______________________________________________________________________
o2::EMCAL::Digit Digitizer::HitToDigit(Hit hit)
{
  TVector3 pos(hit.GetX(), hit.GetY(), hit.GetZ());
  Int_t tower = mGeometry->GetAbsCellIdFromEtaPhi(pos.PseudoRapidity(), pos.Phi());
  Double_t amplitude = hit.GetEnergyLoss();
  Digit digit(tower, amplitude, mEventTime);
  return digit;
}

//_______________________________________________________________________
void Digitizer::setEventTime(double t)
{
  // assign event time, it should be in a strictly increasing order
  // convert to ns
  t *= mCoeffToNanoSecond;

  if (t<mEventTime && mContinuous) {
    LOG(FATAL) << "New event time (" << t << ") is < previous event time (" << mEventTime << ")" << FairLogger::endl;
  }
  mEventTime = t;
  /*
  // to limit the range of RO Frame IDs we subtract the meaningless offset
  if (mContinuous) { // in continuous mode we set the offset only in the very beginning
    if (!mParams.isTimeOffsetSet()) { // offset is initially at -inf
      mParams.setTimeOffset(mEventTime + mParams.getROFrameLenght()*(gRandom->Rndm()-0.5));
    }
  }
  else { // in the triggered mode we start from 0 ROFrame in every event
    mParams.setTimeOffset( mEventTime + mParams.getROFrameLenght()*(gRandom->Rndm()-0.5));
    mROFrameMin = 0;  // so we reset the frame counters
    mROFrameMax = 0;
  }
  */
}

//_______________________________________________________________________
void Digitizer::fillOutputContainer(std::vector<Digit>* digits, UInt_t maxFrame)
{
  /*
  // fill output with digits ready to be stored, generating the noise beforehand
  if (maxFrame>mROFrameMax) maxFrame = mROFrameMax;

  LOG(INFO) << "Filling ITS digits output for RO frames " << mROFrameMin << ":" << maxFrame << FairLogger::endl ;

  for (auto &simulation : mSimulations) {
    // add the random noise to all ROFrame being stored
    simulation.addNoise(mROFrameMin,maxFrame);
  }

  // we have to write chips in RO increasing order, therefore have to loop over the frames here
  for (auto rof=mROFrameMin;rof<=maxFrame;rof++) {
    for (auto &simulation : mSimulations) {
      simulation.fillOutputContainer(digits,rof);
    }
  }
  mROFrameMin = maxFrame+1;
  */
}

//_______________________________________________________________________
void Digitizer::setCurrSrcID(int v)
{
  // set current MC source ID
  if ( v > MCCompLabel::maxSourceID() ) {
    LOG(FATAL) << "MC source id " << v << " exceeds max storable in the label "
	       << MCCompLabel::maxSourceID() << FairLogger::endl;
  }
  mCurrSrcID = v;
}

//_______________________________________________________________________
void Digitizer::setCurrEvID(int v)
{
  // set current MC event ID
  if ( v > MCCompLabel::maxEventID() ) {
    LOG(FATAL) << "MC event id " << v << " exceeds max storable in the label "
	       << MCCompLabel::maxEventID() << FairLogger::endl;
  }
  mCurrEvID = v;
}
