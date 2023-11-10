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

// \file RUInfo.cxx
// \brief Transient structures for ITS and MFT HW -> SW mapping

#include "ITSMFTReconstruction/RUInfo.h"
#include "Framework/Logger.h"

using namespace o2::itsmft;

std::string ChipOnRUInfo::asString() const
{
  return fmt::format("ChonRu:{:3d} ModSW:{:2d} ChOnModSW:{:2d} CabSW:{:3d}| ChOnCab:{:1d} | CabHW:{:2d} | CabPos:{:2d} | ModHW:{:2d} | ChOnModHW:{:2d}",
                     int(id), int(moduleSW), int(chipOnModuleSW), int(cableSW), int(chipOnCable), int(cableHW), int(cableHWPos), int(moduleHW), int(chipOnModuleHW));
}

void ChipOnRUInfo::print() const
{
  LOG(info) << asString();
}

std::string ChipInfo::asString() const
{
  return fmt::format("CH{:5d} RUTyp:{:d} RU:{:3d} | {}", int(id), int(ruType), int(ru), chOnRU ? chOnRU->asString() : std::string{});
}

void ChipInfo::print() const
{
  LOGP(info, fmt::runtime(asString()));
}
