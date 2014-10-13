#ifndef ALICEO2_ITS_CONTAINERFACTORY_H_
#define ALICEO2_ITS_CONTAINERFACTORY_H_

#include "FairContFact.h"

class FairContainer;

namespace AliceO2 {
namespace ITS {

class ContainerFactory : public FairContFact
{
  private:
    void setAllContainers();
  public:
    ContainerFactory();
    ~ContainerFactory() {}
    FairParSet* createContainer(FairContainer*);
    ClassDef( ContainerFactory,0) // Factory for all O2its parameter containers
};
}
}

#endif
