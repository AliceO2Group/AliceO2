#include "MFTSimulation/EventHeader.h"

using namespace o2::MFT;

ClassImp(o2::MFT::EventHeader)

//_____________________________________________________________________________
EventHeader::EventHeader()
: FairEventHeader()
  , mPartNo(0)
{

}

//_____________________________________________________________________________
EventHeader::~EventHeader()
= default;
