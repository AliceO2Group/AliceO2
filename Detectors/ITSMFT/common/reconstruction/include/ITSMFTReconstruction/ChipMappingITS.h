// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CHIPMAPPINGITS_H
#define ALICEO2_CHIPMAPPINGITS_H

// \file ChipMappingITS.h
// \brief ITS chip <-> module mapping

#include <Rtypes.h>
#include <array>

namespace o2
{
namespace ITSMFT
{

struct ITSChipMappingData {
  UShort_t module = 0;      // global module ID
  UChar_t chipInModule = 0; // chip within the module
  ClassDefNV(ITSChipMappingData, 1);
};

struct ITSModuleMappingData {
  UChar_t layer = 0;        // layer id
  UChar_t nChips = 0;       // number of chips
  UShort_t firstChipID = 0; // global id of 1st chip
  ClassDefNV(ITSModuleMappingData, 1);
};

class ChipMappingITS
{
 public:
  static constexpr int getNModules() { return NModules; }
  static constexpr int getNChips() { return NChips; }

  int chipID2Module(int chipID, int& chipInModule) const
  {
    chipInModule = ChipMappingData[chipID].chipInModule;
    return ChipMappingData[chipID].module;
  }

  int chipID2Module(int chipID) const
  {
    return ChipMappingData[chipID].module;
  }

  int getNChipsInModule(int modID) const
  {
    return ModuleMappingData[modID].nChips;
  }

  int module2ChipID(int modID, int chipInModule) const
  {
    return ModuleMappingData[modID].firstChipID + chipInModule;
  }

  int module2Layer(int modID) const
  {
    return ModuleMappingData[modID].layer;
  }

  int chip2Layer(int chipID) const
  {
    return ModuleMappingData[ChipMappingData[chipID].module].layer;
  }

 private:
  static constexpr int NChips = 24120;
  static constexpr int NModules = 1740;

  static const std::array<ITSChipMappingData, NChips> ChipMappingData;
  static const std::array<ITSModuleMappingData, NModules> ModuleMappingData;

  ClassDefNV(ChipMappingITS, 1)
};
}
}

#endif
