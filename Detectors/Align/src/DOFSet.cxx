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

/// @file   DOFSet.cxx
/// @author ruben.shahoyan@cern.ch
/// @brief  Interface to contiguous set of DOFs in the controller class

#include "Align/DOFSet.h"
#include "Align/Controller.h"

using namespace o2::align;

DOFSet::DOFSet(const char* symname, Controller* ctr) : TNamed(symname, ""), mController(ctr)
{
  if (!ctr) {
    LOG(fatal) << "Controller has to be provided :" << symname;
  }
}

//_________________________________________________________
const float* DOFSet::getParVals() const
{
  return &mController->getGloParVal()[mFirstParGloID];
}

//_________________________________________________________
const float* DOFSet::getParErrs() const
{
  return &mController->getGloParErr()[mFirstParGloID];
}

//_________________________________________________________
const int* DOFSet::getParLabs() const
{
  return &mController->getGloParLab()[mFirstParGloID];
}

//_________________________________________________________
float* DOFSet::getParVals()
{
  return &mController->getGloParVal()[mFirstParGloID];
}

//_________________________________________________________
float* DOFSet::getParErrs()
{
  return &mController->getGloParErr()[mFirstParGloID];
}

//_________________________________________________________
int* DOFSet::getParLabs()
{
  return &mController->getGloParLab()[mFirstParGloID];
}
