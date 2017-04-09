/// \file Hit.cxx
/// \brief Implementation of the Hit class
/// \author bogdan.vulpescu@cern.ch 
/// \date 10/10/2016

#include "MFTReconstruction/Hit.h"

#include "TVector3.h"

using namespace o2::MFT;

ClassImp(o2::MFT::Hit)

//_____________________________________________________________________________
Hit::Hit() : 
FairHit()
{

}

//_____________________________________________________________________________
Hit::Hit(Int_t detID, TVector3& pos, TVector3& dpos, Int_t mcindex) : 
  FairHit(detID, pos, dpos, mcindex)
{

}

//_____________________________________________________________________________
Hit::~Hit()
= default;
