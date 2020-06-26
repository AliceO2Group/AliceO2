// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ContainerFactory.cxx
/// \brief Implementation of the ContainerFactory class

#include "ECLayersBase/ContainerFactory.h"
#include "FairRuntimeDb.h" // for FairRuntimeDb
#include "TString.h"       // for TString

class FairParSet;

using namespace o2::ecl;

ClassImp(o2::ecl::ContainerFactory);

static ContainerFactory gO2eclContFact;

ContainerFactory::ContainerFactory() : FairContFact()
{
  fName = "ContainerFactory";
  fTitle = "Factory for parameter containers in libO2ecl";
  mSetAllContainers();
  FairRuntimeDb::instance()->addContFactory(this);
}

void ContainerFactory::mSetAllContainers()
{
  // FairContainer* p= new FairContainer("O2eclGeoPar",
  //                                    "O2ecl Geometry Parameters",
  //                                    "TestDefaultContext");
  // p->addContext("TestNonDefaultContext");
  //
  // containers->Add(p);
}

FairParSet* ContainerFactory::createContainer(FairContainer* c)
{
  // const char* name=c->GetName();
  // FairParSet* p=NULL;
  // if (strcmp(name,"O2eclGeoPar")==0) {
  //  p=new O2eclGeoPar(c->getConcatName().Data(),
  //                          c->GetTitle(),c->getContext());
  //}
  // return p;
  return nullptr;
}
