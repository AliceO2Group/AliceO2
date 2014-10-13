#include "O2itsContFact.h"
#include "FairRuntimeDb.h"

#include <iostream>

ClassImp(O2itsContFact)

static O2itsContFact gO2itsContFact;

O2itsContFact::O2itsContFact()
  : FairContFact()
{
  /** Constructor (called when the library is loaded) */
  fName="O2itsContFact";
  fTitle="Factory for parameter containers in libO2its";
  setAllContainers();
  FairRuntimeDb::instance()->addContFactory(this);
}

void O2itsContFact::setAllContainers()
{
  /** Creates the Container objects with all accepted
      contexts and adds them to
      the list of containers for the O2its library.
  */
/*
  FairContainer* p= new FairContainer("O2itsGeoPar",
                                      "O2its Geometry Parameters",
                                      "TestDefaultContext");
  p->addContext("TestNonDefaultContext");

  containers->Add(p);
*/
 }

FairParSet* O2itsContFact::createContainer(FairContainer* c)
{
  /** Calls the constructor of the corresponding parameter container.
      For an actual context, which is not an empty string and not
      the default context
      of this container, the name is concatinated with the context.
  */
 /* const char* name=c->GetName();
  FairParSet* p=NULL;
  if (strcmp(name,"O2itsGeoPar")==0) {
    p=new O2itsGeoPar(c->getConcatName().Data(),
                            c->GetTitle(),c->getContext());
  }
  return p;
*/
  }
