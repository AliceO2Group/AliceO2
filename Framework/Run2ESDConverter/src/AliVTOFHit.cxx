#include "AliVTOFHit.h"

ClassImp(AliVTOFHit)

//________________________________________
AliVTOFHit & AliVTOFHit::operator=(const AliVTOFHit& source)
{
  // assignment op-r
  if (this == &source) return *this;
  TObject::operator=(source);
  return *this;
}

