/// \file Point.cxx
/// \brief Implementation of the Point class

#include "ITSMFTSimulation/Point.h"

#include <iostream>

ClassImp(AliceO2::ITSMFT::Point)

using std::cout;
using std::endl;
using namespace AliceO2::ITSMFT;

Point::Point() : FairMCPoint(),
                 mTrackStatus(0),
                 mTrackStatusStart(0),
                 mShunt(0),
                 mStartX(0.),
                 mStartY(0.),
                 mStartZ(0.),
                 mStartTime(0.),
                 mTotalEnergy(0.)
{
}

Point::Point(Int_t trackID, Int_t detID, TVector3 startPos, TVector3 pos, TVector3 mom,
             Double_t startTime, Double_t time, Double_t length, Double_t eLoss, Int_t shunt, Int_t status,
             Int_t statusStart)
  : FairMCPoint(trackID, detID, pos, mom, time, length, eLoss),
    mTrackStatus(status),
    mTrackStatusStart(statusStart),
    mShunt(shunt),
    mStartX(startPos.X()),
    mStartY(startPos.Y()),
    mStartZ(startPos.Z()),
    mStartTime(startTime),
    mTotalEnergy(0.)
{
}

Point::~Point()
= default;

void Point::Print(const Option_t *opt) const
{
  cout << *this;
}


