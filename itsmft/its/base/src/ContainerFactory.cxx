/// \file ContainerFactory.cxx
/// \brief Implementation of the ContainerFactory class

#include "include/ContainerFactory.h"
#include "FairRuntimeDb.h"  // for FairRuntimeDb
#include "TString.h"        // for TString
class FairParSet;

using namespace AliceO2::ITS;

ClassImp(AliceO2::ITS::ContainerFactory)

static ContainerFactory gO2itsContFact;

ContainerFactory::ContainerFactory() : FairContFact()
{
  fName = "ContainerFactory";
  fTitle = "Factory for parameter containers in libO2its";
  mSetAllContainers();
  FairRuntimeDb::instance()->addContFactory(this);
}

void ContainerFactory::mSetAllContainers()
{
  // FairContainer* p= new FairContainer("O2itsGeoPar",
  //                                    "O2its Geometry Parameters",
  //                                    "TestDefaultContext");
  // p->addContext("TestNonDefaultContext");
  //
  // containers->Add(p);
}

FairParSet* ContainerFactory::createContainer(FairContainer* c)
{
  // const char* name=c->GetName();
  // FairParSet* p=NULL;
  // if (strcmp(name,"O2itsGeoPar")==0) {
  //  p=new O2itsGeoPar(c->getConcatName().Data(),
  //                          c->GetTitle(),c->getContext());
  //}
  // return p;
  return 0;
}
