#include "MFTBase/EventHeader.h"

using namespace AliceO2::MFT;

ClassImp(AliceO2::MFT::EventHeader)

//_____________________________________________________________________________
EventHeader::EventHeader()
: FairEventHeader()
  , fPartNo(0)
{

}

//_____________________________________________________________________________
EventHeader::~EventHeader()
{

}
