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
  UChar_t chipOnModule = 0; // chip within the module
  UChar_t cable = 0;        // cable in the connector
  UChar_t chipOnRU = 0;     // chip within the RU (SW)
  ClassDefNV(MFTChipMappingData, 1);
};

struct MFTModuleMappingData {
  UChar_t layer = 0;        // layer id
  UChar_t nChips = 0;       // number of chips
  UShort_t firstChipID = 0; // global id of 1st chip
  UChar_t connector = 0;    // cable connector in a zone
  UChar_t zone = 0;         // read-out zone id
  UChar_t disk = 0;         // disk id
  UChar_t half = 0;         // half id
  ClassDefNV(MFTModuleMappingData, 1);
};

class ChipMappingMFT
{
 public:
  ChipMappingMFT();
  ~ChipMappingMFT() = default;

  static constexpr std::string_view getName() { return "MFT"; }

  // RS placeholder for methods to implement ----------->

  ///< total number of RUs
  static constexpr Int_t getNRUs() { return NRUs; }

  ///< get FEEId of the RU (software id of the RU), read via given link
  uint8_t FEEId2RUSW(uint16_t hw) const { return mFEEId2RUSW[hw]; }

  ///< get HW id of the RU (software id of the RU)
  uint16_t RUSW2FEEId(uint16_t sw, uint16_t linkID = 0) const { return mRUInfo[sw].idHW; }

  ///< compose FEEid for given stave (ru) relative to layer and link, see documentation in the constructor
  uint16_t composeFEEId(uint16_t layer, uint16_t ruOnLayer, uint16_t link) const
  {
    // only one link is used
    // ruOnLayer is 0, 1, 2, 3 for half = 0
    //              4, 5, 6, 7            1
    auto dhalf = std::div(ruOnLayer, 4);
    uint16_t half = dhalf.quot;
    uint16_t zone = dhalf.rem;
    auto ddisk = std::div(layer, 2);
    uint16_t disk = ddisk.quot;
    uint16_t plane = layer % 2;
    return (half << 6) + (disk << 3) + (plane << 2) + zone;
  }

  ///< decompose FEEid to layer, stave (ru) relative to layer, link, see documentation in the constructor
  void expandFEEId(uint16_t feeID, uint16_t& layer, uint16_t& ruOnLayer, uint16_t& link) const
  {
    link = 0;
    uint16_t half = feeID >> 6;
    uint16_t disk = (feeID >> 3) & 0x7;
    uint16_t plane = (feeID >> 2) & 0x1;
    uint16_t zone = feeID & 0x3;
    layer = 2 * disk + plane;
    ruOnLayer = 2 * half + zone;
  }

  ///< get info on sw RU
  const RUInfo* getRUInfoFEEId(Int_t feeID) const { return &mRUInfo[FEEId2RUSW(feeID)]; }

  ///< get number of chips served by single cable on given RU type
  uint8_t getGBTHeaderRUType(Int_t ruType, Int_t cableHW)
  {
    return (cableHW & 0x1f);
  }

  ///< convert HW cable ID to SW ID for give RU type
  uint8_t cableHW2SW(uint8_t ruType, uint8_t hwid) const { return mCableHW2SW[ruType][hwid]; }

  ///< get chip global SW ID from chipID on module, cable SW ID and stave (RU) info
  uint16_t getGlobalChipID(uint16_t chOnModuleHW, Int_t cableHW, const RUInfo& ruInfo) const
  {
    return ruInfo.firstChipIDSW + mCableHWFirstChip[ruInfo.ruType][cableHW] + chipModuleIDHW2SW(ruInfo.ruType, chOnModuleHW);
  }

  ///< convert HW id of chip in the module to SW ID (sequential ID on the module)
  int chipModuleIDHW2SW(int ruType, int hwIDinMod) const
  {
    return hwIDinMod;
  }

  ///< convert SW id of chip in the module to HW ID
  int chipModuleIDSW2HW(int ruType, int swIDinMod) const
  {
    return swIDinMod;
  }

  static constexpr Int_t getNChips() { return NChips; }

  static constexpr Int_t getNModules() { return NModules; }

  Int_t chipID2Module(Int_t chipID, Int_t& chipOnModule) const
  {
    chipOnModule = ChipMappingData[chipID].chipOnModule;
    return ChipMappingData[chipID].module;
  }

  Int_t chipID2Module(Int_t chipID) const
  {
    return ChipMappingData[chipID].module;
  }

  Int_t getNChipsInModule(Int_t modID) const
  {
    return ModuleMappingData[modID].nChips;
  }

  Int_t module2ChipID(Int_t modID, Int_t chipOnModule) const
  {
    return ModuleMappingData[modID].firstChipID + chipOnModule;
  }

  Int_t module2Layer(Int_t modID) const
  {
    return ModuleMappingData[modID].layer;
  }

  Int_t chip2Layer(Int_t chipID) const
  {
    return ModuleMappingData[ChipMappingData[chipID].module].layer;
  }

  ///< impose user defined FEEId -> ruSW (staveID) conversion, to be used only for forced decoding of corrupted data
  void imposeFEEId2RUSW(uint16_t, uint16_t) {}

  ///< extract information about the chip with SW ID
  void getChipInfoSW(Int_t chipSW, ChipInfo& chInfo) const
  {
    UShort_t ladder = ChipMappingData[chipSW].module;
    UChar_t layer = ModuleMappingData[ladder].layer;
    UChar_t zone = ModuleMappingData[ladder].zone;
    UChar_t half = ModuleMappingData[ladder].half;

    chInfo.ruType = ZoneRUType[zone][layer / 2];

    // count RU SW per half layers
    //chInfo.ru = NLayers * (NZonesPerLayer / 2) * half + (NZonesPerLayer / 2) * layer + zone;

    // count RU SW per full layers
    chInfo.ru = NZonesPerLayer * layer + (NZonesPerLayer / 2) * half + zone;

    chInfo.id = ChipMappingData[chipSW].chipOnRU;
    chInfo.chOnRU = getChipOnRUInfo(chInfo.ruType, chInfo.id);
  }

  ///< get number of chips served by RU of given type (i.e. RU type for ITS)
  Int_t getNChipsOnRUType(Int_t ruType) const { return NChipsOnRUType[ruType]; }

  /// < extract information about the chip properties on the stave of given type for the chip
  /// < with sequential ID SWID within the stave
  const ChipOnRUInfo* getChipOnRUInfo(Int_t ruType, Int_t chOnRUSW) const
  {
    return &mChipsInfo[mChipInfoEntryRU[ruType] + chOnRUSW];
  }

  static constexpr std::int16_t getRUDetectorField() { return 0x0; }

  ///< get pattern of lanes on the RU served by a given RU type
  Int_t getCablesOnRUType(Int_t ruType) const { return (0x1 << NChipsOnRUType[ruType]) - 1; }

  ///< get info on sw RU
  const RUInfo* getRUInfoSW(int ruSW) const { return &mRUInfo[ruSW]; }

  ///< convert layer ID and RU sequential ID on Layer to absolute RU IDSW
  int getRUIDSW(int layer, int ruOnLayer) const
  {
    int sid = 0;
    for (int i = 0; i < NLayers; i++) {
      if (i >= layer)
        break;
      sid += NZonesPerLayer;
    }
    return sid + ruOnLayer;
  }

 private:
  Int_t invalid() const;
  static constexpr Int_t NZonesPerLayer = 2 * 4;
  static constexpr Int_t NLayers = 10;
  static constexpr Int_t NRUs = NLayers * NZonesPerLayer;
  static constexpr Int_t NModules = 280;
  static constexpr Int_t NChips = 936;
  static constexpr Int_t NRUTypes = 12;
  static constexpr Int_t NChipsInfo = 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 16 + 17 + 18 + 19;
  static constexpr Int_t NChipsPerCable = 1;
  static constexpr Int_t NLinks = 1;
  static constexpr Int_t NConnectors = 5;
  static constexpr Int_t NMaxChipsPerLadder = 5;
  static constexpr Int_t NRUCables = 25;

  static constexpr Int_t ZoneLadderIDmin[NZonesPerLayer / 2][NLayers]{
    { 0, 21, 0, 21, 0, 23, 0, 28, 0, 29 },
    { 3, 18, 3, 18, 3, 20, 4, 24, 5, 25 },
    { 6, 15, 6, 15, 6, 17, 8, 20, 9, 21 },
    { 9, 12, 9, 12, 9, 13, 12, 16, 13, 17 }
  };
  static constexpr Int_t ZoneLadderIDmax[NZonesPerLayer / 2][NLayers]{
    { 2, 23, 2, 23, 2, 25, 3, 31, 4, 33 },
    { 5, 20, 5, 20, 5, 22, 7, 27, 8, 28 },
    { 8, 17, 8, 17, 8, 19, 11, 23, 12, 24 },
    { 11, 14, 11, 14, 12, 16, 15, 19, 16, 20 }
  };

  static constexpr Int_t ZoneRUType[NZonesPerLayer / 2][NLayers / 2]{
    { 1, 1, 1, 7, 11 },
    { 2, 2, 4, 8, 9 },
    { 2, 2, 3, 8, 10 },
    { 0, 0, 5, 6, 7 }
  };

  static constexpr Int_t ChipConnectorCable[NConnectors][NMaxChipsPerLadder]{
    { 5, 6, 7, 24, 23 },
    { 0, 1, 2, 3, 4 },
    { 17, 16, 15, 14, 13 },
    { 22, 21, 20, 19, 18 },
    { 12, 11, 10, 9, 8 }
  };

  static const std::array<MFTChipMappingData, NChips> ChipMappingData;
  static const std::array<MFTModuleMappingData, NModules> ModuleMappingData;

  ///< number of chips per zone (RU)
  static constexpr std::array<int, NRUTypes> NChipsOnRUType{ 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19 };

  // info on chips info within the zone (RU)
  std::array<ChipOnRUInfo, NChipsInfo> mChipsInfo;
  Int_t mChipInfoEntryRU[NRUTypes];

  /// info per zone (RU)
  std::array<RUInfo, NRUs> mRUInfo;
  std::vector<uint8_t> mFEEId2RUSW; // HW RU ID -> SW ID conversion

  std::vector<uint8_t> mCableHW2SW[NRUs];       ///< table of cables HW to SW conversion for each RU type
  std::vector<uint8_t> mCableHWFirstChip[NRUs]; ///< 1st chip of module (relative to the 1st chip of the stave) served by each cable

  ClassDefNV(ChipMappingMFT, 1)
};
} // namespace itsmft
} // namespace o2

#endif
