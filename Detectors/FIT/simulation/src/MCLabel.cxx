// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MCLabel 
/// \brief Implementation of the MC true container
#include "FITSimulation/MCLabel.h"

using namespace o2::fit;

ClassImp(o2::fit::MCLabel);

MCLabel::MCLabel(Int_t trackID, Int_t eventID, Int_t srcID, Int_t qID)
  : o2::MCCompLabel(trackID, eventID, srcID),
    mDetID(qID)
{

  //  std::cout<<"@@@ MCLabel constructor "<<std::endl;
}
