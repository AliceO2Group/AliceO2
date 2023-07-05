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

#include <DetectorsBase/SimFieldUtils.h>
#include <Field/MagneticField.h>
#include <Field/ALICE3MagneticField.h>
#include <SimConfig/SimConfig.h>
#include <CCDB/BasicCCDBManager.h>
#include <DataFormatsParameters/GRPMagField.h>

using namespace o2::base;

FairField* const SimFieldUtils::createMagField()
{
  if (getenv("ALICE3_SIM_FIELD")) {
    return new o2::field::ALICE3MagneticField();
  }

  auto& confref = o2::conf::SimConfig::Instance();
  // a) take field from CDDB
  const auto fieldmode = confref.getConfigData().mFieldMode;
  o2::field::MagneticField* field = nullptr;
  if (fieldmode == o2::conf::SimFieldMode::kCCDB) {
    LOG(info) << "Fetching magnetic field from CCDB";
    auto& ccdb = o2::ccdb::BasicCCDBManager::instance();
    auto grpmagfield = ccdb.get<o2::parameters::GRPMagField>("GLO/Config/GRPMagField");
    // TODO: clarify if we need to pass other params such as beam energy/type etc.
    field = o2::field::MagneticField::createFieldMap(grpmagfield->getL3Current(), grpmagfield->getDipoleCurrent(), grpmagfield->getFieldUniformity());
  }
  // b) using the given values on the command line
  else {
    field = o2::field::MagneticField::createNominalField(confref.getConfigData().mField, confref.getConfigData().mFieldMode == o2::conf::SimFieldMode::kUniform);
  }
  return field;
}
