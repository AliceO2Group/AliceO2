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

#ifndef ALICEO2_MFT_CHIPMAPPING_H
#define ALICEO2_MFT_CHIPMAPPING_H

// \file ChipMappingMFT.h
// \brief MFT chip <-> module mapping

#include <Rtypes.h>
#include <array>
#include <FairLogger.h>
#include "Headers/DataHeader.h"
#include "ITSMFTReconstruction/RUInfo.h"
#include "DetectorsCommonDataFormats/DetID.h"

namespace o2
{
namespace itsmft
{

struct MFTChipMappingData {
  UShort_t module = 0;         // global module ID
  UChar_t chipOnModule = 0;    // chip within the module
  UChar_t cable = 0;           // cable in the connector
  UChar_t chipOnRU = 0;        // chip within the RU (SW)
  UShort_t globalChipSWID = 0; // global software chip ID
  UShort_t localChipSWID = 0;  // local software chip ID
  UShort_t localChipHWID = 0;  // local hardware chip ID
  UChar_t connector = 0;       // cable connector in a zone
  UChar_t zone = 0;            // read-out zone id
  UChar_t ruOnLayer = 0;       // read-out-unit index on layer
  UChar_t ruType = 0;          // read-out-unit type
  UChar_t ruSWID = 0;          // read-out-unit hardware ID
  UChar_t ruHWID = 0;          // read-out-unit software ID
  UChar_t layer = 0;           // MFT layer
  UChar_t disk = 0;            // MFT disk
  UChar_t half = 0;            // MFT half
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
  static constexpr o2::header::DataOrigin getOrigin() { return o2::header::gDataOriginMFT; }
  static constexpr o2::detectors::DetID::ID getDetID() { return o2::detectors::DetID::MFT; }

  // RS placeholder for methods to implement ----------->

  ///< total number of RUs
  static constexpr Int_t getNRUs() { return NRUs; }

  ///< get FEEId of the RU (software id of the RU), read via given link
  uint8_t FEEId2RUSW(uint16_t hw) const { return mFEEId2RUSW[hw & 0xff]; }

  ///< get HW id of the RU (software id of the RU)
  uint16_t RUSW2FEEId(uint16_t sw, uint16_t linkID = 0) const { return ((linkID << 8) + mRUInfo[sw].idHW); }

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
    return (link << 8) + (half << 6) + (disk << 3) + (plane << 2) + zone;
  }

  ///< decompose FEEid to layer, stave (ru) relative to layer, link, see documentation in the constructor
  void expandFEEId(uint16_t feeID, uint16_t& layer, uint16_t& ruOnLayer, uint16_t& link) const
  {
    link = feeID >> 8;
    uint16_t half = (feeID >> 6) & 0x1;
    uint16_t disk = (feeID >> 3) & 0x7;
    uint16_t plane = (feeID >> 2) & 0x1;
    uint16_t zone = feeID & 0x3;
    layer = 2 * disk + plane;
    ruOnLayer = 4 * half + zone;
  }

  ///< decompose FEEid to face, disk, half
  void expandFEEIdFaceDiskHalf(uint16_t feeID, uint16_t& face, uint16_t& disk, uint16_t& half) const
  {
    half = (feeID >> 6) & 0x1;
    disk = (feeID >> 3) & 0x7;
    face = (feeID >> 2) & 0x1;
  }

  ///< get info on sw RU
  const RUInfo* getRUInfoFEEId(Int_t feeID) const { return &mRUInfo[FEEId2RUSW(feeID)]; }

  ///< get number of chips served by single cable on given RU type
  uint8_t getGBTHeaderRUType(Int_t ruType, Int_t cableHW)
  {
    return ((0x1 << 7) + (cableHW & 0x1f));
  }

  ///< convert HW cable ID to its position on the ActiveLanes word in the GBT.header for given RU type
  uint8_t cableHW2Pos(uint8_t ruType, uint8_t hwid) const { return mCableHW2Pos[ruType][hwid]; }

  ///< convert HW cable ID to SW ID for give RU type
  uint8_t cableHW2SW(uint8_t ruType, uint8_t hwid) const { return mCableHW2SW[ruType][hwid]; }

  ///< convert cable iterator ID to its position on the ActiveLanes word in the GBT.header for given RU type
  uint8_t cablePos(uint8_t ruType, uint8_t id) const { return mCablePos[ruType][id]; }

  ///< get chip global SW ID from chipID on module, cable SW ID and stave (RU) info
  uint16_t getGlobalChipID(uint16_t chOnModuleHW, int cableHW, const RUInfo& ruInfo) const
  {
    auto chipOnRU = cableHW2SW(ruInfo.ruType, cableHW);
    return mRUGlobalChipID[(int)(ruInfo.idSW)].at((int)(chipOnRU));
  }

  ///< convert HW id of chip in the module to SW ID (sequential ID on the module)
  int chipModuleIDHW2SW(int ruType, int hwIDinMod) const
  {
    return (8 - hwIDinMod);
  }

  ///< convert SW id of chip in the module to HW ID
  int chipModuleIDSW2HW(int ruType, int swIDinMod) const
  {
    return (8 - swIDinMod);
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
    // ladder (MFT) = module (ITS)
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
  const ChipOnRUInfo* getChipOnRUInfo(Int_t ruType, Int_t chipOnRU) const
  {
    return &mChipsInfo[mChipInfoEntryRU[ruType] + chipOnRU];
  }

  static constexpr std::int16_t getRUDetectorField() { return 0x0; }

  uint32_t getCablesOnRUType(Int_t ruType) const
  {
    uint32_t pattern = 0;
    for (Int_t i = 0; i < NRUCables; i++) {
      pattern |= (0x1 << mCableHW2Pos[ruType][i]);
    }
    return pattern;
  }

  ///< get info on sw RU
  const RUInfo* getRUInfoSW(int ruSW) const { return &mRUInfo[ruSW]; }

  ///< convert layer ID and RU sequential ID on Layer to absolute RU IDSW
  int getRUIDSW(int layer, int ruOnLayer) const
  {
    int sid = 0;
    for (int i = 0; i < NLayers; i++) {
      if (i >= layer) {
        break;
      }
      sid += NZonesPerLayer;
    }
    return sid + ruOnLayer;
  }

  const int getNZonesPerLayer() const { return NZonesPerLayer; }
  const int getNLayers() const { return NLayers; }

  ///< convert zone number [0...8] and layer number [0...10] to RU type
  int getRUType(int zone, int layer) const { return ZoneRUType[zone % 4][layer / 2]; }

  static constexpr int NChips = 936, NLayers = 10, NZonesPerLayer = 2 * 4, NRUTypes = 13;

  const std::array<MFTChipMappingData, NChips>& getChipMappingData() const { return ChipMappingData; }

  const auto& getModuleMappingData() const { return ModuleMappingData; }

  void print() const;

  ///< LayerID of each MFT chip
  static constexpr std::array<int, NChips> ChipID2Layer{
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9};

  static constexpr std::array<int, 280> mLadderIDGeoToRO{
    0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 5, 24, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 25, 26, 27, 28, 39, 40, 41, 42, 43, 44, 45,
    46, 47, 29, 48, 52, 53, 66, 67, 54, 55, 56, 68, 69, 57, 58,
    49, 50, 59, 60, 70, 71, 61, 62, 63, 72, 73, 64, 65, 51, 74,
    75, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 76, 77, 78,
    79, 80, 81, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 82,
    83, 106, 107, 114, 115, 132, 133, 116, 117, 118, 119, 120, 134, 135, 121,
    122, 108, 109, 110, 111, 123, 124, 136, 137, 125, 126, 127, 128, 129, 138,
    139, 130, 131, 112, 113, 140, 146, 147, 148, 149, 150, 151, 152, 153, 154,
    141, 142, 143, 144, 155, 156, 157, 158, 159, 160, 161, 162, 163, 145, 164,
    170, 171, 172, 173, 174, 175, 176, 177, 178, 165, 166, 167, 168, 179, 180,
    181, 182, 183, 184, 185, 186, 187, 169, 188, 192, 193, 206, 207, 194, 195,
    196, 208, 209, 197, 198, 189, 190, 199, 200, 210, 211, 201, 202, 203, 212,
    213, 204, 205, 191, 214, 215, 224, 225, 226, 227, 228, 229, 230, 231, 232,
    233, 234, 216, 217, 218, 219, 220, 221, 235, 236, 237, 238, 239, 240, 241,
    242, 243, 244, 245, 222, 223, 246, 247, 254, 255, 272, 273, 256, 257, 258,
    259, 260, 274, 275, 261, 262, 248, 249, 250, 251, 263, 264, 276, 277, 265,
    266, 267, 268, 269, 278, 279, 270, 271, 252, 253};

  static constexpr std::array<int, NChips> mChipIDGeoToRO{
    0, 1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 2,
    3, 4, 5, 6, 7, 8, 9, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 10, 11, 66, 67, 78, 79, 80, 81, 82, 83, 84,
    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 102, 103, 104, 68, 69, 70, 71, 72, 73, 74, 75, 105, 106,
    107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
    122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 76, 77, 132, 133, 140,
    141, 142, 143, 144, 145, 182, 183, 184, 185, 186, 187, 188, 189, 146, 147,
    148, 149, 150, 151, 152, 153, 154, 190, 191, 192, 193, 194, 195, 196, 197,
    155, 156, 157, 158, 159, 160, 134, 135, 136, 137, 161, 162, 163, 164, 165,
    166, 198, 199, 200, 201, 202, 203, 204, 205, 167, 168, 169, 170, 171, 172,
    173, 174, 175, 206, 207, 208, 209, 210, 211, 212, 213, 176, 177, 178, 179,
    180, 181, 138, 139, 214, 215, 216, 217, 218, 219, 244, 245, 246, 247, 248,
    249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
    264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278,
    279, 280, 281, 282, 283, 284, 285, 286, 287, 220, 221, 222, 223, 224, 225,
    226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 288, 289, 290,
    291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305,
    306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320,
    321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 238, 239, 240, 241,
    242, 243, 332, 333, 334, 335, 336, 337, 356, 357, 358, 359, 360, 361, 362,
    363, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 364, 365, 366, 367,
    368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382,
    383, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 384, 385, 386, 387,
    388, 389, 390, 391, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348,
    349, 392, 393, 394, 395, 396, 397, 398, 399, 448, 449, 450, 451, 452, 453,
    454, 455, 456, 457, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410,
    411, 412, 413, 414, 415, 416, 417, 418, 419, 458, 459, 460, 461, 462, 463,
    464, 465, 466, 467, 420, 421, 422, 423, 424, 425, 426, 427, 350, 351, 352,
    353, 354, 355, 468, 469, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489,
    490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504,
    505, 506, 470, 471, 472, 473, 474, 475, 476, 477, 507, 508, 509, 510, 511,
    512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526,
    527, 528, 529, 530, 531, 532, 533, 478, 479, 534, 535, 546, 547, 548, 549,
    550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564,
    565, 566, 567, 568, 569, 570, 571, 572, 536, 537, 538, 539, 540, 541, 542,
    543, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586,
    587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 544, 545,
    600, 601, 608, 609, 610, 611, 612, 613, 650, 651, 652, 653, 654, 655, 656,
    657, 614, 615, 616, 617, 618, 619, 620, 621, 622, 658, 659, 660, 661, 662,
    663, 664, 665, 623, 624, 625, 626, 627, 628, 602, 603, 604, 605, 629, 630,
    631, 632, 633, 634, 666, 667, 668, 669, 670, 671, 672, 673, 635, 636, 637,
    638, 639, 640, 641, 642, 643, 674, 675, 676, 677, 678, 679, 680, 681, 644,
    645, 646, 647, 648, 649, 606, 607, 682, 683, 684, 685, 686, 687, 712, 713,
    714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728,
    729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743,
    744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 688, 689, 690,
    691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705,
    756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770,
    771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785,
    786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 706,
    707, 708, 709, 710, 711, 800, 801, 802, 803, 804, 805, 824, 825, 826, 827,
    828, 829, 830, 831, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 832,
    833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847,
    848, 849, 850, 851, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 852,
    853, 854, 855, 856, 857, 858, 859, 806, 807, 808, 809, 810, 811, 812, 813,
    814, 815, 816, 817, 860, 861, 862, 863, 864, 865, 866, 867, 916, 917, 918,
    919, 920, 921, 922, 923, 924, 925, 868, 869, 870, 871, 872, 873, 874, 875,
    876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 926, 927, 928,
    929, 930, 931, 932, 933, 934, 935, 888, 889, 890, 891, 892, 893, 894, 895,
    818, 819, 820, 821, 822, 823};

 private:
  Int_t invalid() const;
  static constexpr Int_t NRUs = NLayers * NZonesPerLayer;
  static constexpr Int_t NModules = 280;
  static constexpr Int_t NChipsInfo = 7 + 8 + 9 + 10 + 11 + 12 + 13 + 14 + 16 + 17 + 18 + 19 + 14;
  static constexpr Int_t NChipsPerCable = 1;
  static constexpr Int_t NLinks = 1;
  static constexpr Int_t NConnectors = 5;
  static constexpr Int_t NMaxChipsPerLadder = 5;
  static constexpr Int_t NRUCables = 25;

  static constexpr Int_t ZoneLadderIDmin[NZonesPerLayer / 2][NLayers]{
    {0, 21, 0, 21, 0, 23, 0, 28, 0, 29},
    {3, 18, 3, 18, 3, 20, 4, 24, 5, 25},
    {6, 15, 6, 15, 6, 17, 8, 20, 9, 21},
    {9, 12, 9, 12, 9, 13, 12, 16, 13, 17}};
  static constexpr Int_t ZoneLadderIDmax[NZonesPerLayer / 2][NLayers]{
    {2, 23, 2, 23, 2, 25, 3, 31, 4, 33},
    {5, 20, 5, 20, 5, 22, 7, 27, 8, 28},
    {8, 17, 8, 17, 8, 19, 11, 23, 12, 24},
    {11, 14, 11, 14, 12, 16, 15, 19, 16, 20}};

  static constexpr Int_t ZoneRUType[NZonesPerLayer / 2][NLayers / 2]{
    {1, 1, 1, 7, 11},
    {2, 2, 4, 8, 9},
    {2, 2, 3, 8, 10},
    {0, 0, 5, 6, 12}};

  static constexpr Int_t ChipConnectorCable[NConnectors][NMaxChipsPerLadder]{
    {5, 6, 7, 24, 23},
    {0, 1, 2, 3, 4},
    {17, 16, 15, 14, 13},
    {22, 21, 20, 19, 18},
    {12, 11, 10, 9, 8}};

  static const std::array<MFTChipMappingData, NChips> ChipMappingData;
  static const std::array<MFTModuleMappingData, NModules> ModuleMappingData;

  ///< number of chips per zone (RU)
  static constexpr std::array<int, NRUTypes> NChipsOnRUType{7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 14};

  // info on chips info within the zone (RU)
  std::array<ChipOnRUInfo, NChipsInfo> mChipsInfo;
  Int_t mChipInfoEntryRU[NRUTypes];

  /// info per zone (RU)
  std::array<RUInfo, NRUs> mRUInfo;
  std::vector<uint8_t> mFEEId2RUSW; // HW RU ID -> SW ID conversion

  std::vector<uint8_t> mCableHW2SW[NRUs];       ///< table of cables HW to SW conversion for each RU type
  std::vector<uint8_t> mCableHW2Pos[NRUs];      ///< table of cables positions in the ActiveLanes mask for each RU type
  std::vector<uint8_t> mCablePos[NRUs];         ///< reverse table of cables positions in the ActiveLanes mask for each RU type
  std::vector<uint8_t> mCableHWFirstChip[NRUs]; ///< 1st chip of module (relative to the 1st chip of the stave) served by each cable

  std::array<std::vector<uint16_t>, NRUs> mRUGlobalChipID;

  ClassDefNV(ChipMappingMFT, 1)
};
} // namespace itsmft
} // namespace o2

#endif
