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

/// \file GRPMagField.cxx
/// \brief Implementation of General Run Parameters object for MagField
/// \author ruben.shahoyan@cern.ch

#include <Framework/Logger.h>
#include "DataFormatsParameters/GRPMagField.h"
#include "CommonUtils/NameConf.h"
#include <cstdint>

using namespace o2::parameters;

//_______________________________________________
GRPMagField* GRPMagField::loadFrom(const std::string& grpMagFieldFileName)
{
  // load object from file
  auto fname = o2::base::NameConf::getGRPMagFieldFileName(grpMagFieldFileName);
  TFile flGRPMagField(fname.c_str());
  if (flGRPMagField.IsZombie()) {
    LOG(error) << "Failed to open " << fname;
    throw std::runtime_error("Failed to open GRP Mag Field file");
  }
  auto grpMagField = reinterpret_cast<o2::parameters::GRPMagField*>(flGRPMagField.GetObjectChecked(o2::base::NameConf::CCDBOBJECT.data(), Class()));
  if (!grpMagField) {
    throw std::runtime_error(fmt::format("Failed to load GRP Mag Field object from {}", fname));
  }
  return grpMagField;
}

void GRPMagField::print() const
{
  printf("magnet currents (A) L3 = %.3f, Dipole = %.f; uniformity = %s\n", getL3Current(), getDipoleCurrent(), mUniformField ? "true" : "false");
}

o2::units::Current_t GRPMagField::checkDipoleOverride()
{
  static float v = getenv("O2_OVERRIDE_DIPOLE_CURRENT") ? atof(getenv("O2_OVERRIDE_DIPOLE_CURRENT")) : NOOVERRIDEVAL;
  static bool alarmShown = false;
  if (v != NOOVERRIDEVAL && !alarmShown) {
    LOGP(error, "Overriding DIPOLE current to {}", v);
    alarmShown = true;
  }
  return v;
}

o2::units::Current_t GRPMagField::checkL3Override()
{
  static float v = getenv("O2_OVERRIDE_L3_CURRENT") ? atof(getenv("O2_OVERRIDE_L3_CURRENT")) : NOOVERRIDEVAL;
  static bool alarmShown = false;
  if (v != NOOVERRIDEVAL && !alarmShown) {
    LOGP(error, "Overriding L3 current to {}", v);
    alarmShown = true;
  }
  return v;
}
