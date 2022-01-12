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

#include <FairLogger.h>
#include "DataFormatsParameters/GRPMagField.h"
#include "CommonUtils/NameConf.h"

using namespace o2::parameters;

//_______________________________________________
GRPMagField* GRPMagField::loadFrom(const std::string& grpMagFieldFileName, const std::string& grpMagFieldName)
{
  // load object from file
  auto fname = o2::base::NameConf::getGRPMagFieldFileName(grpMagFieldFileName);
  TFile flGRPMagField(fname.c_str());
  if (flGRPMagField.IsZombie()) {
    LOG(error) << "Failed to open " << fname;
    throw std::runtime_error("Failed to open GRP Mag Field file");
  }
  auto grpMagField = reinterpret_cast<o2::parameters::GRPMagField*>(
    flGRPMagField.GetObjectChecked(grpMagFieldName.data(), o2::parameters::GRPMagField::Class()));
  if (!grpMagField) {
    LOG(error) << "Did not find GRP Mag Field object named " << grpMagFieldName;
    throw std::runtime_error("Failed to load GRP Mag Field object");
  }
  return grpMagField;
}
