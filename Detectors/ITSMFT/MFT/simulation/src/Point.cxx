/// \file Point.cxx
/// \brief Implementation of the Point class
/// \author bogdan.vulpescu@cern.ch 
/// \date 10/10/2016

#include "MFTSimulation/Point.h"

using namespace o2::MFT;

ClassImp(o2::MFT::Point)

//_____________________________________________________________________________
Point::Point() : 
FairMCPoint()
{

}

//_____________________________________________________________________________
Point::Point(Int_t trackID, Int_t detID, TVector3 pos, TVector3 mom, Double_t tof, Double_t length, Double_t eLoss) : 
FairMCPoint(trackID, detID, pos, mom, tof, length, eLoss)
{

}

//_____________________________________________________________________________
Point::~Point()
= default;

