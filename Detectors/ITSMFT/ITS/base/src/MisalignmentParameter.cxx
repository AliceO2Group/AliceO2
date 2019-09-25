// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MisalignmentParameter.cxx
/// \brief Implementation of the MisalignmentParameter class

#include "ITSBase/MisalignmentParameter.h"

#include "FairParamList.h"

using namespace o2::its;

ClassImp(o2::its::MisalignmentParameter);

MisalignmentParameter::MisalignmentParameter(const char* name, const char* title, const char* context)
  : FairParGenericSet(name, title, context),
    mShiftX(),
    mShiftY(),
    mShiftZ(),
    mRotX(),
    mRotY(),
    mRotZ(),
    mNumberOfDetectors(0)
{
}

MisalignmentParameter::~MisalignmentParameter() = default;
void MisalignmentParameter::Clear() {}
void MisalignmentParameter::putParams(FairParamList* list)
{
  if (!list) {
    return;
  }

  list->add("NumberOfDetectors", mNumberOfDetectors);
  list->add("ShiftX", mShiftX);
  list->add("ShiftY", mShiftY);
  list->add("ShiftZ", mShiftZ);
  list->add("RotationX", mRotX);
  list->add("RotationY", mRotY);
  list->add("RotationZ", mRotZ);
}

Bool_t MisalignmentParameter::getParams(FairParamList* list)
{
  if (!list) {
    return kFALSE;
  }

  if (!list->fill("NumberOfDetectors", &mNumberOfDetectors)) {
    return kFALSE;
  }

  mShiftX.Set(mNumberOfDetectors);
  if (!list->fill("ShiftX", &mShiftX)) {
    return kFALSE;
  }

  mShiftY.Set(mNumberOfDetectors);
  if (!list->fill("ShiftY", &mShiftY)) {
    return kFALSE;
  }

  mShiftZ.Set(mNumberOfDetectors);
  if (!list->fill("ShiftZ", &mShiftZ)) {
    return kFALSE;
  }

  mRotX.Set(mNumberOfDetectors);
  if (!list->fill("RotationX", &mRotX)) {
    return kFALSE;
  }

  mRotY.Set(mNumberOfDetectors);
  if (!list->fill("RotationY", &mRotY)) {
    return kFALSE;
  }

  mRotZ.Set(mNumberOfDetectors);
  if (!list->fill("RotationZ", &mRotZ)) {
    return kFALSE;
  }

  return kTRUE;
}
