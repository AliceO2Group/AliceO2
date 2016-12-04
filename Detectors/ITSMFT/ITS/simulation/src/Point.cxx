/// \file Point.cxx
/// \brief Implementation of the Point class

#include "ITSSimulation/Point.h"

#include <iostream>

using std::cout;
using std::endl;
using namespace AliceO2::ITS;

Point::Point() : FairMCPoint(),
                 fTrackStatus(0),
                 fTrackStatusStart(0),
                 fShunt(0),
                 fStartX(0.),
                 fStartY(0.),
                 fStartZ(0.),
                 fStartTime(0.),
                 fTotalEnergy(0.)
{
}

Point::Point(Int_t trackID, Int_t detID, TVector3 startPos, TVector3 pos, TVector3 mom,
             Double_t startTime, Double_t time, Double_t length, Double_t eLoss, Int_t shunt, Int_t status,
             Int_t statusStart)
  : FairMCPoint(trackID, detID, pos, mom, time, length, eLoss),
    fTrackStatus(status),
    fTrackStatusStart(statusStart),
    fShunt(shunt),
    fStartX(startPos.X()),
    fStartY(startPos.Y()),
    fStartZ(startPos.Z()),
    fStartTime(startTime),
    fTotalEnergy(0.)
{
}

Point::~Point()
{
}

void Point::Print(const Option_t *opt) const
{
  cout << *this;
}


ClassImp(AliceO2::ITS::Point)
