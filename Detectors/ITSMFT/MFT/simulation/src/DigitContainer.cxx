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
