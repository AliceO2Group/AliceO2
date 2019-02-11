#include "TObject.h"

#include "AliESDTrdTrigger.h"

AliESDTrdTrigger::AliESDTrdTrigger() : 
  TObject()
{
  // default ctor

  for (Int_t iSector = 0; iSector < fgkNsectors; iSector++) {
    fFlags[iSector] = 0x0;
  }
}

AliESDTrdTrigger::AliESDTrdTrigger(const AliESDTrdTrigger &rhs) :
  TObject(rhs)
{
  // copy ctor

  for (Int_t iSector = 0; iSector < fgkNsectors; iSector++) {
    fFlags[iSector] = rhs.fFlags[iSector];
  }
}

AliESDTrdTrigger& AliESDTrdTrigger::operator=(const AliESDTrdTrigger &rhs)
{
  // assignment operator
  if (&rhs != this) {
    TObject::operator=(rhs);
    for (Int_t iSector = 0; iSector < fgkNsectors; iSector++) {
      fFlags[iSector] = rhs.fFlags[iSector];
    }
  }

  return *this;
}

AliESDTrdTrigger::~AliESDTrdTrigger()
{
  // dtor

}
