// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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

