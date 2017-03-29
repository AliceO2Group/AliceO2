#include "MFTSimulation/EventHeader.h"

using namespace AliceO2::MFT;

ClassImp(AliceO2::MFT::EventHeader)

//_____________________________________________________________________________
EventHeader::EventHeader()
: FairEventHeader()
  , mPartNo(0)
{

}

//_____________________________________________________________________________
EventHeader::~EventHeader()
= default;
