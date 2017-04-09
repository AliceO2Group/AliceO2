
#include "TPCBase/ContainerFactory.h"
#include "FairRuntimeDb.h"  // for FairRuntimeDb

class FairParSet;
using namespace o2::TPC;

ClassImp(ContainerFactory)

static ContainerFactory gTpcContainerFactory;

ContainerFactory::ContainerFactory()
  : FairContFact()
{
  /** Constructor (called when the library is loaded) */
  fName="ContainerFactory";
  fTitle="Factory for parameter containers in libO2tpc";
  setAllContainers();
  FairRuntimeDb::instance()->addContFactory(this);
}

void ContainerFactory::setAllContainers()
{
  /** Creates the Container objects with all accepted
      contexts and adds them to
      the list of containers for the O2tpc library.
  */

 /* FairContainer* p= new FairContainer("O2tpcGeoPar",
                                      "O2tpc Geometry Parameters",
                                      "TestDefaultContext");
  p->addContext("TestNonDefaultContext");

  containers->Add(p);
*/
}

FairParSet* ContainerFactory::createContainer(FairContainer* c)
{
  /** Calls the constructor of the corresponding parameter container.
      For an actual context, which is not an empty string and not
      the default context
      of this container, the name is concatinated with the context.
  */
    /*
  const char* name=c->GetName();
  FairParSet* p=NULL;
  if (strcmp(name,"O2tpcGeoPar")==0) {
    p=new O2tpcGeoPar(c->getConcatName().Data(),
                            c->GetTitle(),c->getContext());
  }
  return p;
     */
  return nullptr;
}
