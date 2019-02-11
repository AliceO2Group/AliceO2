#include "AliVMultiplicity.h"

ClassImp(AliVMultiplicity)

void AliVMultiplicity::Clear(Option_t* )
{
  // !!! Don't clear TNamed part: name is used to search the object
  TObject::Clear();
}
