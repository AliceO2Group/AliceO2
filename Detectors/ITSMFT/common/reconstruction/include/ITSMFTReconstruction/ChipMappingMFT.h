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
#include "ITSMFTReconstruction/RUInfo.h"

namespace o2
{
namespace itsmft
{

struct MFTChipMappingData {
  UShort_t module = 0;      // global module ID
  UChar_t chipInModule = 0; // chip within the module
  ClassDefNV(MFTChipMappingData, 1);
};

struct MFTModuleMappingData {
  UChar_t layer = 0;        // layer id
  UChar_t nChips = 0;       // number of chips
  UShort_t firstChipID = 0; // global id of 1st chip
  ClassDefNV(MFTModuleMappingData, 1);
};

class ChipMappingMFT
{
 public:
  static constexpr std::string_view getName() { return "MFT"; }

  // RS placeholder for methods to implement ----------->

  ///< total number of RUs
  static constexpr int getNRUs() { return 0; }

  ///< get FEEId of the RU (software id of the RU), read via given link
  uint8_t FEEId2RUSW(uint16_t hw) const { return 0; }

  ///< get HW id of the RU (software id of the RU)
  uint16_t RUSW2FEEId(uint16_t sw, uint16_t linkID = 0) const { return 0; }

  ///< compose FEEid for given stave (ru) relative to layer and link, see documentation in the constructor
  uint16_t composeFEEId(uint16_t lr, uint16_t ruOnLr, uint16_t link) const { return 0; }

  ///< decompose FEEid to layer, stave (ru) relative to layer, link, see documentation in the constructor
  void expandFEEId(uint16_t feeID, uint16_t& lr, uint16_t& ruOnLr, uint16_t& link) const
  {
    lr = ruOnLr = link = 0;
  }

  ///< get info on sw RU
  const RUInfo* getRUInfoSW(int ruSW) const { return nullptr; }

  ///< get info on sw RU
  const RUInfo* getRUInfoFEEId(int feeID) const { return nullptr; }

  ///< get number of chips served by single cable on given RU type
  uint8_t getGBTHeaderRUType(int ruType, int cableHW) { return 0; }

  ///< convert HW cable ID to SW ID for give RU type
  uint8_t cableHW2SW(uint8_t ruType, uint8_t hwid) const { return 0; }

  ///< get chip global SW ID from chipID on module, cable SW ID and stave (RU) info
  uint16_t getGlobalChipID(uint16_t chOnModuleHW, int cableHW, const RUInfo& ruInfo) const
  {
    return 0;
  }

  static constexpr int getNChips() { return NChips; }

  // RS placeholder for methods to implement -----------<

  static constexpr int getNModules() { return NModules; }

  int chipID2Module(int chipID, int& chipInModule) const
  {
    chipInModule = ChipMappingData[chipID].chipInModule;
    return ChipMappingData[chipID].module;
  }

  int chipID2Module(int chipID) const
  {
    return ChipMappingData[chipID].module;
  }

  int getCablesOnRUType(int ruType) const { return 0; }

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

  ///< impose user defined FEEId -> ruSW (staveID) conversion, to be used only for forced decoding of corrupted data
  void imposeFEEId2RUSW(uint16_t, uint16_t) {}

 private:
  int invalid() const;
  static constexpr int NModules = 280;
  static constexpr int NChips = 920;

  static const std::array<MFTChipMappingData, NChips> ChipMappingData;
  static const std::array<MFTModuleMappingData, NModules> ModuleMappingData;

  ClassDefNV(ChipMappingMFT, 1)
};
}
}

#endif
