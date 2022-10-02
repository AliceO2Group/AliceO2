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

#include "Field/MagFieldFact.h"
#include "FairField.h"
#include <fairlogger/Logger.h>
#include "FairRuntimeDb.h"
#include "Field/MagFieldParam.h"
#include "Field/MagneticField.h"

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

using namespace o2::field;

ClassImp(MagFieldFact);

static MagFieldFact gMagFieldFact;

MagFieldFact::MagFieldFact() : FairFieldFactory(), mFieldPar(nullptr) { fCreator = this; }
MagFieldFact::~MagFieldFact() = default;

void MagFieldFact::SetParm()
{
  auto RunDB = FairRuntimeDb::instance();
  mFieldPar = (MagFieldParam*)RunDB->getContainer("MagFieldParam");
}

FairField* MagFieldFact::createFairField()
{
  if (!mFieldPar) {
    LOG(error) << "MagFieldFact::createFairField: No field parameters available";
    return nullptr;
  }
  // since we have just 1 field class, we don't need to consider fFieldPar->GetType()
  std::cerr << "creating the field as unmanaged pointer\n";
  // we have to use naked "new" here since the FairRootAna or TGeoGlobalMagField expect
  // bare pointer on the field which they eventually destroy
  return new MagneticField(*mFieldPar);
}
