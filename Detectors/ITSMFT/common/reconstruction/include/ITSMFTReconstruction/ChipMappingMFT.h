// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MFT_CHIPMAPPING_H
#define ALICEO2_MFT_CHIPMAPPING_H

// \file ChipMappingMFT.h
// \brief MFT chip <-> module mapping

#include <Rtypes.h>
#include <array>

namespace o2
{
namespace ITSMFT
{

class ChipMappingMFT
{
 public:
  static constexpr int getNModules() { return NModules; }
  static constexpr int getNChips() { return NChips; }

  int chipID2Module(int chipID, int& chipInModule) const
  {
    chipInModule = -1;
    return invalid();
  }

  int chipID2Module(int chipID) const
  {
    return invalid();
  }

  int getNChipsInModule(int modID) const
  {
    return invalid();
  }

  int module2ChipID(int modID, int chipInModule) const
  {
    return invalid();
  }

  int module2Layer(int modID) const
  {
    return invalid();
  }

  int chip2Layer(int chipID) const
  {
    return invalid();
  }

 private:
  int invalid() const;
  static constexpr int NModules = -1;
  static constexpr int NChips = 920;

  ClassDefNV(ChipMappingMFT, 1)
};
}
}

#endif
