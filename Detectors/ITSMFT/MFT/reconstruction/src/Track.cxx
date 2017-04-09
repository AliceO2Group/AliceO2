/// \file Track.cxx
/// \brief Implementation of the Track class
/// \author bogdan.vulpescu@cern.ch 
/// \date 11/10/2016

#include "MFTReconstruction/Track.h"

using namespace o2::MFT;

ClassImp(o2::MFT::Track)

//_____________________________________________________________________________
Track::Track() : 
FairTrackParam()
{

}

//_____________________________________________________________________________
Track::~Track()
= default;

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

