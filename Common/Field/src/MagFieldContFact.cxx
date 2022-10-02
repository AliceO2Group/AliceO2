// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MagFieldContFact.cxx
/// \brief Implementation of the MagFieldContFact: factory for ALICE mag. field
/// \author ruben.shahoyan@cern.ch

#include <cstring>         // for strcmp, NULL
#include <fairlogger/Logger.h> // for FairLogger
#include "FairRuntimeDb.h" // for FairRuntimeD
#include "FairParSet.h"
#include "Field/MagFieldParam.h" // for FairConstPar
#include "Field/MagFieldContFact.h"

using namespace o2::field;

ClassImp(MagFieldContFact) static MagFieldContFact gMagFieldContFact;

MagFieldContFact::MagFieldContFact() : FairContFact()
{
  // Constructor (called when the library is loaded)
  fName = "MagFieldContFact";
  fTitle = "Factory for MagField parameter container";
  setAllContainers();
  FairRuntimeDb::instance()->addContFactory(this);
}

void MagFieldContFact::setAllContainers()
{
  //  Creates the Container objects and adds it to the list of containers

  auto* p = new FairContainer("MagFieldParam", "Mag. Field Parameters", "Default Field");
  containers->Add(p);
}

FairParSet* MagFieldContFact::createContainer(FairContainer* c)
{
  // calls the constructor of the corresponding parameter container.
  const char* name = c->GetName();
  LOG(info) << "MagFieldContFact::createContainer: Creating mag.field container " << name;
  FairParSet* p = nullptr;
  if (strcmp(name, "MagFieldParam") == 0) {
    p = new MagFieldParam(c->getConcatName().Data(), c->GetTitle(), c->getContext());
  }
  return p;
}
