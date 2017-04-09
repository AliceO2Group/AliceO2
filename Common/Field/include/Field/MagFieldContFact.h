/// \file MagFieldContFact.h
/// \brief Definition of the MagFieldContFact: factory for ALICE mag. field
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_FIELD_MAGFIELDCONTFACT_
#define ALICEO2_FIELD_MAGFIELDCONTFACT_

#include "FairContFact.h"
#include "Rtypes.h" // for ClassDef

class FairParSet;

namespace o2 {
  namespace Field {


class MagFieldParam;

class  MagFieldContFact : public FairContFact
{
  private:
    void setAllContainers();

  public:
    MagFieldContFact();
    ~MagFieldContFact() = default;
    FairParSet* createContainer(FairContainer*);
    
    ClassDef(MagFieldContFact, 0) // Factory for Magnetic field parameters containers
};

}
}
#endif
