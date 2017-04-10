#ifndef ALICEO2_TPC_CONTAINERFACTORY_H_
#define ALICEO2_TPC_CONTAINERFACTORY_H_

#include "FairContFact.h"  // for FairContFact, FairContainer (ptr only)
#include "Rtypes.h"        // for ContainerFactory::Class, ClassDef, etc
class FairParSet;

class FairContainer;

namespace o2 {
namespace TPC {



class ContainerFactory : public FairContFact
{
  private:
    void setAllContainers();
  public:
    ContainerFactory();
    ~ContainerFactory() override = default;
    FairParSet* createContainer(FairContainer*) override;
    ClassDefOverride( o2::TPC::ContainerFactory,0) // Factory for all tpc parameter containers
};
}
}
#endif
