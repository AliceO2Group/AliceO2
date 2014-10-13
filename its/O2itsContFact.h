#ifndef O2ITSCONTFACT_H
#define O2ITSCONTFACT_H

#include "FairContFact.h"

class FairContainer;

class O2itsContFact : public FairContFact
{
  private:
    void setAllContainers();
  public:
    O2itsContFact();
    ~O2itsContFact() {}
    FairParSet* createContainer(FairContainer*);
    ClassDef( O2itsContFact,0) // Factory for all O2its parameter containers
};

#endif
