// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// \file RUInfo.cxx
// \brief Transient structures for ITS and MFT HW -> SW mapping

#include <ITSMFTReconstruction/RUInfo.h>

using namespace o2::itsmft;

void ChipOnRUInfo::print() const
{
  printf("ChonRu:%3d ModSW:%2d ChOnModSW:%2d CabSW:%3d| ChOnCab:%d | CabHW: ",
         id, moduleSW, chipOnModuleSW, cableSW, chipOnCable);
  for (int i = 8; i--;) {
    printf("%d", (cableHW & (0x1 << i)) ? 1 : 0);
  }
  printf(" | ModHW ");
  for (int i = 8; i--;) {
    printf("%d", (moduleHW & (0x1 << i)) ? 1 : 0);
  }
  printf(" | ChOnModHW: %2d\n", chipOnModuleHW);
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
