// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitContainer.h
/// \brief Container of digits
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#include "ITSMFTBase/Digit.h"

#include "MFTSimulation/DigitContainer.h"

using o2::ITSMFT::Digit;

using namespace o2::MFT;

//_____________________________________________________________________________
void DigitContainer::reset()
{

}

//_____________________________________________________________________________
Digit* DigitContainer::getDigit() 
{ 

  return nullptr; 

}

//_____________________________________________________________________________
Digit* DigitContainer::addDigit()
{

  return nullptr;

}

//_____________________________________________________________________________
void DigitContainer::fillOutputContainer(TClonesArray* output)
{

}
