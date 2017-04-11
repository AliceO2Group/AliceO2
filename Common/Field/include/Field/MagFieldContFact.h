/// \file MagFieldContFact.h
/// \brief Definition of the MagFieldContFact: factory for ALICE mag. field
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_FIELD_MAGFIELDCONTFACT_
#define ALICEO2_FIELD_MAGFIELDCONTFACT_

#include "FairContFact.h"
#include "Rtypes.h" // for ClassDef

class FairParSet;

namespace o2 {
  namespace field {


class MagFieldParam;

class  MagFieldContFact : public FairContFact
{
  private:
    void setAllContainers();

  public:
    MagFieldContFact();
    ~MagFieldContFact() override = default;
    FairParSet* createContainer(FairContainer*) override;
    
    ClassDefOverride(MagFieldContFact, 0) // Factory for Magnetic field parameters containers
};

}
}
#endif
