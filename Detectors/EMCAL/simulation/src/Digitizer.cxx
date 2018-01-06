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
#include <forward_list>
#include <climits>

ClassImp(o2::EMCAL::Digitizer);

//using o2::EMCAL::GeometryBase;
using o2::EMCAL::Hit;
using o2::EMCAL::Digit;

using namespace o2::EMCAL;

//_______________________________________________________________________
void Digitizer::init()
{
}

//_______________________________________________________________________
void Digitizer::finish()
{
}

//_______________________________________________________________________
void Digitizer::process(const std::vector<Hit>* hits, std::vector<Digit>* digits)
{
  digits->clear();
  mDigits.clear();

  for(auto hit : *hits) {
    Digit digit = HitToDigit(hit);
    Int_t id = digit.GetTower();

    if(id<0 || id>mGeometry->GetNCells()){
      LOG(WARNING) << "tower index out of range: " << id << FairLogger::endl;
      continue;
    }

    Bool_t flag=false;
    for(auto rit=mDigits[id].rbegin(); rit!=mDigits[id].rend(); ++rit){
      if(rit->CanAdd(digit)) {
	(*rit) += digit;
	flag = true;
	break;
      }
    }
    if(!flag) mDigits[id].push_back(digit);
  }

  fillOutputContainer(digits);
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
}

//_______________________________________________________________________
void Digitizer::fillOutputContainer(std::vector<Digit>* digits)
{
  std::forward_list<Digit> l;

  for (auto tower : mDigits){
    for(auto it=tower.second.rbegin(); it!=tower.second.rend(); ++it) l.push_front(*it);
  }

  //testing

  l.sort();

  for(auto digit : l) digits->push_back(digit);
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
