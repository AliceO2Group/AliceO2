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
#include <bitset>

using namespace o2::itsmft;

void ChipOnRUInfo::print() const
{
  std::bitset<8> chw(cableHW), mhw(moduleHW);
  printf("ChonRu:%3d ModSW:%2d ChOnModSW:%2d CabSW:%3d| ChOnCab:%d | CabHW: %8s (%2d) | ModHW: %8s | ChOnModHW: %2d\n",
         id, moduleSW, chipOnModuleSW, cableSW, chipOnCable, chw.to_string().c_str(), cableHWPos,
         mhw.to_string().c_str(), chipOnModuleHW);
}

void ChipInfo::print() const
{
  printf("CH%5d RUTyp:%d RU:%3d | ", id, ruType, ru);
  if (chOnRU) {
    chOnRU->print();
  } else {
    printf("\n");
  }
}
