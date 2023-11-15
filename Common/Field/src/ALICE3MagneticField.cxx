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

/// \file ALICE3MagneticField.cxx
/// \brief Implementation of the ALICE3 MagF class
/// \author sandro.wenzel@cern.ch

#include <Field/ALICE3MagneticField.h>
#include <fairlogger/Logger.h>                    // for FairLogger
#include <CommonUtils/ConfigurationMacroHelper.h> // for just-in-time compilation of macros
#include <filesystem>

using namespace o2::field;
// ClassImp(ALICE3MagneticField)

// initializes the just-in-time implementation of the field function
void ALICE3MagneticField::initJITFieldFunction()
{
  // for now we check if there is an env variable, pointing to a macro file
  auto filename = getenv("ALICE3_MAGFIELD_MACRO");
  if (filename) {
    LOG(info) << "Taking ALICE3 magnetic field implementation from macro (just in time)";
    if (std::filesystem::exists(filename)) {
      // if this file exists we will compile the hook on the fly
      mJITFieldFunction = o2::conf::GetFromMacro<FieldEvalFcn>(filename, "field()", "function<void(const double*,double*)>", "o2mc_alice3_field_hook");
      LOG(info) << "Hook initialized from file " << filename;
    } else {
      LOG(error) << "Did not find file " << filename;
    }
  }
}

void ALICE3MagneticField::init()
{
  LOG(info) << "Initializing ALICE3 magnetic field";
  initJITFieldFunction();
}

void ALICE3MagneticField::Field(const Double_t* __restrict__ xyz, Double_t* __restrict__ b)
{
  if (mJITFieldFunction) {
    mJITFieldFunction(xyz, b);
  } else {
    // TODO: These values are just toy; Real implementation should go here
    b[0] = 0.;
    b[1] = 0.;
    b[2] = -10.; // -10 kGauss
  }
}
