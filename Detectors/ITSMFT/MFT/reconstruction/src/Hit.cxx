// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
