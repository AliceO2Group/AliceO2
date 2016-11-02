/// \file Track.cxx
/// \brief Implementation of the Track class
/// \author bogdan.vulpescu@cern.ch 
/// \date 11/10/2016

#include "MFTReconstruction/Track.h"

using namespace AliceO2::MFT;

ClassImp(AliceO2::MFT::Track)

//_____________________________________________________________________________
Track::Track() : 
FairTrackParam()
{

}

//_____________________________________________________________________________
Track::~Track()
{

}

//_____________________________________________________________________________
Track::Track(const Track& track) :
  FairTrackParam(track)
{

  *this = track;

}

//_____________________________________________________________________________
Track& Track::operator=(const Track& track) 
{

  return *this;

}

