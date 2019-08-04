// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MagFieldParam.cxx
/// \brief Implementation of the MagFieldParam class
/// \author ruben.shahoyan@cern.ch

#include "Field/MagFieldParam.h"
#include "Field/MagneticField.h"
#include "FairParamList.h"
#include "FairRun.h"
#include "FairRuntimeDb.h"

using namespace o2::field;

//========================================
ClassImp(MagFieldParam);

MagFieldParam::MagFieldParam(const char* name, const char* title, const char* context)
  : FairParGenericSet(name, title, context), mMapType(k5kG), mBeamType(kNoBeamField), mDefaultIntegration(0), mFactorSol(0.), mFactorDip(0.), mBeamEnergy(0.), mMaxField(0.), mMapPath()
{
  /// create param for alice mag. field
}

void MagFieldParam::SetParam(const MagneticField* field)
{
  /// fill parameters from the initialized field
  //  SetName(field->GetName()); ? is this needed
  //  SetTitle(field->GetTitle());
  mMapType = field->getMapType();
  mBeamType = field->getBeamType();
  mDefaultIntegration = field->Integral();
  mFactorSol = field->getFactorSolenoid();
  mFactorDip = field->getFactorDipole();
  mBeamEnergy = field->getBeamEnergy();
  mMaxField = field->Max();
  mMapPath = field->getDataFileName();
  //
}

void MagFieldParam::putParams(FairParamList* list)
{
  /// store parameters in the list
  if (!list)
    return;
  list->add("Map  Type  ID", int(mMapType));
  list->add("Beam Type  ID", int(mBeamType));
  list->add("Integral Type", mDefaultIntegration);
  list->add("Fact.Solenoid", mFactorSol);
  list->add("Fact.Dipole  ", mFactorDip);
  list->add("Beam Energy  ", mBeamEnergy);
  list->add("Max. Field   ", mMaxField);
  list->add("Path to map  ", mMapPath.Data());
  //
}

Bool_t MagFieldParam::getParams(FairParamList* list)
{
  /// retried parameters
  int int2enum = 0;
  if (!list->fill("Map  Type  ID", &int2enum))
    return kFALSE;
  mMapType = static_cast<BMap_t>(int2enum);
  if (!list->fill("Beam Type  ID", &int2enum))
    return kFALSE;
  mBeamType = static_cast<BeamType_t>(int2enum);
  //
  if (!list->fill("Integral Type", &mDefaultIntegration))
    return kFALSE;
  if (!list->fill("Fact.Solenoid", &mFactorSol))
    return kFALSE;
  if (!list->fill("Fact.Dipole  ", &mFactorDip))
    return kFALSE;
  if (!list->fill("Beam Energy  ", &mBeamEnergy))
    return kFALSE;
  if (!list->fill("Max. Field   ", &mMaxField))
    return kFALSE;
  FairParamObj* parpath = list->find("Path to map  ");
  if (!parpath)
    return kFALSE;
  int lgt = parpath->getLength();
  // RS: is there a bug in FairParamList::fill(const Text_t* name,Text_t* value,const Int_t length)?
  // I think the "if (l<length-1)" should be "if (l<length)"
  char cbuff[lgt + 2];
  memset(cbuff, 0, sizeof(char) * (lgt + 2));
  if (!list->fill("Path to map  ", cbuff, lgt + 2))
    return kFALSE;
  mMapPath = cbuff;
  return kTRUE;
}
