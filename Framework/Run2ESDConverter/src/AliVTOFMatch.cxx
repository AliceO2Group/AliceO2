#include "AliVTOFMatch.h"

ClassImp(AliVTOFMatch)

//___________________________________________________
AliVTOFMatch & AliVTOFMatch::operator=(const AliVTOFMatch& source) 
{
  // assignment op-r
  if (this == &source) return *this;
  TObject::operator=(source);
  return *this;
}
