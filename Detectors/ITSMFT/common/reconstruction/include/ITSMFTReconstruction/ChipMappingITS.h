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
#include <cstdlib>
#include <cstdint>
#include <string>
#include "ITSMFTReconstruction/RUInfo.h"

namespace o2
{
namespace itsmft
{

class ChipMappingITS
{
 public:
  ///< these public methods must be defined in the mapping class for raw data encoding/decoding
  ChipMappingITS();
  ~ChipMappingITS() = default;

  static constexpr std::string_view getName() { return "ITS"; }

  const std::vector<uint8_t>& getCableHWFirstChip(int s) const { return mCableHWFirstChip[s]; }

  static constexpr std::int16_t getRUDetectorField() { return 0x0; }

  ///< total number of RUs
  static constexpr int getNRUs() { return NStavesSB[IB] + NStavesSB[MB] + NStavesSB[OB]; }

  ///< total number of chips
  static constexpr int getNChips() { return NChipsSB[IB] + NChipsSB[MB] + NChipsSB[OB]; }

  ///< number of staves on layer
  static constexpr int getNStavesOnLr(int l) { return NStavesOnLr[l]; }

  ///<  first staves of layer
  static constexpr int getFirstStavesOnLr(int l) { return FirstStaveOnLr[l]; }

  ///< numbes of chips per layer
  static constexpr int getNChipsPerLr(int l) { return NStavesOnLr[l] * NChipsPerStaveSB[RUTypeLr[l]]; }

  ///< compose FEEid for given stave (ru) relative to layer and link, see documentation in the constructor
  uint16_t composeFEEId(uint16_t lr, uint16_t ruOnLr, uint16_t link) const { return (lr << 12) + (link << 8) + (ruOnLr); }

  ///< decompose FEEid to layer, stave (ru) relative to layer, link, see documentation in the constructor
  void expandFEEId(uint16_t feeID, uint16_t& lr, uint16_t& ruOnLr, uint16_t& link) const
  {
    lr = feeID >> 12;
    ruOnLr = feeID & 0x3f;
    link = (feeID >> 8) & 0x3;
  }

  ///< impose user defined FEEId -> ruSW (staveID) conversion, to be used only for forced decoding of corrupted data
  void imposeFEEId2RUSW(uint16_t feeID, uint16_t ruSW);

  ///< modify linkID field in FEEId
  uint16_t modifyLinkInFEEId(uint16_t feeID, uint16_t linkID) const
  {
    feeID &= ~(0x3 << 8);
    feeID |= (0x3 & linkID) << 8;
    return feeID;
  }

  ///< expand SW chip ID to SW (continuous) id's for layer, stave, substave etc.
  void expandChipInfoSW(int idSW, int& lay, int& sta, int& ssta, int& mod, int& chipInMod) const;

  ///< expand SW chip ID to HW  id's for layer, stave, substave, module, chipOnModule
  void expandChipInfoHW(int idSW, int& lay, int& sta, int& ssta, int& mod, int& chipInMod) const;

  ///< convert global SW chip ID to name in HW conventions
  std::string getChipNameHW(int idSW) const;

  void print() const;

  /// < extract information about the chip properties on the stave of given type for the chip
  /// < with sequential ID SWID within the stave
  const ChipOnRUInfo* getChipOnRUInfo(int staveType, int chOnRUSW) const
  {
    return &mChipsInfo[mChipInfoEntrySB[staveType] + chOnRUSW];
  }

  ///< extract information about the chip with SW ID
  void getChipInfoSW(int chipSW, ChipInfo& chInfo) const
  {
    chInfo.id = chipSW;
    if (chipSW > NChipsSB[IB] + NChipsSB[MB] - 1) { // OB
      chipSW -= NChipsSB[IB] + NChipsSB[MB];
      chInfo.ruType = OB;
      auto dvRU = std::div(chipSW, NChipsPerStaveSB[OB]);
      chInfo.ru = NStavesSB[IB] + NStavesSB[MB] + dvRU.quot; // RU ID == stave ID
      chInfo.chOnRU = getChipOnRUInfo(OB, dvRU.rem);
    } else if (chipSW > NChipsSB[IB] - 1) { // MB
      chipSW -= NChipsSB[IB];
      chInfo.ruType = MB;
      auto dvRU = std::div(chipSW, NChipsPerStaveSB[MB]);
      chInfo.ru = NStavesSB[IB] + dvRU.quot; // RU ID == stave ID
      chInfo.chOnRU = getChipOnRUInfo(MB, dvRU.rem);
    } else { // IB
      chInfo.ruType = IB;
      auto dvRU = std::div(chipSW, NChipsPerStaveSB[IB]);
      chInfo.ru = dvRU.quot; // RU ID == stave ID = module ID
      chInfo.chOnRU = getChipOnRUInfo(IB, dvRU.rem);
    }
  }

  ///< get chip global SW ID from chipID on module, cable SW ID and stave (RU) info
  uint16_t getGlobalChipID(uint16_t chOnModuleHW, int cableHW, const RUInfo& ruInfo) const
  {
    return ruInfo.firstChipIDSW + mCableHWFirstChip[ruInfo.ruType][cableHW] + chipModuleIDHW2SW(ruInfo.ruType, chOnModuleHW);
  }

  ///< get SW id of the RU from RU HW id
  uint8_t FEEId2RUSW(uint16_t hw) const { return mFEEId2RUSW[hw]; }

  ///< get FEEId of the RU (software id of the RU), read via given link
  uint16_t RUSW2FEEId(uint16_t sw, uint16_t linkID = 0) const
  {
    uint16_t feeID = mStavesInfo[sw].idHW; // this is valid for link 0 only
    return linkID ? modifyLinkInFEEId(feeID, linkID) : feeID;
  }

  ///< get layer of the RU (from the software id of the RU)
  uint16_t RUSW2Layer(uint16_t sw) const { return mStavesInfo[sw].layer; }

  ///< get layer of the RU (from the software id of the RU)
  uint16_t RUSW2RUType(uint16_t sw) const { return mStavesInfo[sw].ruType; }

  ///< get info on sw RU
  const RUInfo* getRUInfoSW(int ruSW) const { return &mStavesInfo[ruSW]; }

  ///< get info on sw RU
  const RUInfo* getRUInfoFEEId(int feeID) const { return &mStavesInfo[FEEId2RUSW(feeID)]; }

  ///< get number of chips served by single cable on given RU type
  uint8_t getGBTHeaderRUType(int ruType, int cableHW) { return GBTHeaderFlagSB[ruType] + (cableHW & 0x1f); }

  ///< convert HW cable ID to SW ID for give RU type
  uint8_t cableHW2SW(uint8_t ruType, uint8_t hwid) const { return mCableHW2SW[ruType][hwid]; }

  ///< get number of chips served by single cable on given RU type
  int getNChipsPerCable(int ruType) { return NChipsPerCableSB[ruType]; }

  ///< get number cables on the RU served by a given RU type
  int getNCablesOnRUType(int ruType) const { return NCablesPerStaveSB[ruType]; }

  ///< get pattern of lanes on the RU served by a given RU type
  int getCablesOnRUType(int ruType) const { return CablesOnStaveSB[ruType]; }

  ///< get number of chips served by RU of given type (i.e. RU type for ITS)
  int getNChipsOnRUType(int ruType) const { return NChipsPerStaveSB[ruType]; }

  ///< get RU type from the sequential ID of the RU
  int getRUType(int ruID) const
  {
    ///< get the RU type corresponding to RU with secuential number ruID
    if (ruID > NStavesSB[IB] + NStavesSB[MB] - 1) {
      return OB;
    }
    if (ruID > NStavesSB[IB] - 1) {
      return MB;
    }
    return IB;
  }

  ///< convert HW id of chip in the module to SW ID (sequential ID on the module)
  int chipModuleIDHW2SW(int ruType, int hwIDinMod) const
  {
    return ruType == IB ? hwIDinMod : ChipOBModHW2SW[hwIDinMod];
  }

  ///< convert SW id of chip in the module to HW ID
  int chipModuleIDSW2HW(int ruType, int swIDinMod) const
  {
    return ruType == IB ? swIDinMod : ChipOBModSW2HW[swIDinMod];
  }

  ///< convert layer ID and RU sequential ID on Layer to absolute RU IDSW
  int getRUIDSW(int lr, int ruOnLr) const
  {
    int sid = 0;
    for (int i = 0; i < NLayers; i++) {
      if (i >= lr)
        break;
      sid += NStavesOnLr[i];
    }
    return sid + ruOnLr;
  }

 private:
  // sub-barrel types, their number, N layers, Max N GBT Links per RU
  static constexpr int IB = 0, MB = 1, OB = 2, NSubB = 3, NLayers = 7, NLinks = 3;

  static constexpr std::array<uint8_t, NSubB> GBTHeaderFlagSB = {0x1 << 5, 0x1 << 6, 0x1 << 6}; // prefixes for data GBT header byte

  ///< N chips per cable of each sub-barrel
  static constexpr std::array<int, NSubB> NChipsPerCableSB = {1, 7, 7};

  ///< N modules along the stave (or halfstave..)
  static constexpr std::array<int, NSubB> NModulesAlongStaveSB = {9, 4, 7};

  ///< N chips per module of each sub-barrel
  static constexpr std::array<int, NSubB> NChipsPerModuleSB = {9, 14, 14};

  ///< N cables per module of each sub-barrel
  static constexpr std::array<int, NSubB> NCablesPerModule = {9, 2, 2}; // NChipsPerModuleSB[]/NChipsPerCableSB[]

  ///< N modules per stage of each sub-barrel
  static constexpr std::array<int, NSubB> NModulesPerStaveSB = {1, 8, 14};

  ///< number of staves per layer
  static constexpr std::array<int, NLayers> NStavesOnLr = {12, 16, 20, 24, 30, 42, 48};

  ///< number of staves per layer
  static constexpr std::array<int, NLayers> FirstStaveOnLr = {0, 12, 28, 48, 72, 102, 144};

  ///< RU types for each layer
  static constexpr std::array<uint8_t, NLayers> RUTypeLr = {IB, IB, IB, MB, MB, OB, OB};

  ///< number of staves per sub-barrel
  static constexpr std::array<int, NSubB> NStavesSB = {NStavesOnLr[0] + NStavesOnLr[1] + NStavesOnLr[2],
                                                       NStavesOnLr[3] + NStavesOnLr[4],
                                                       NStavesOnLr[5] + NStavesOnLr[6]};
  ///< number of chips per stave of sub-barrel
  static constexpr std::array<int, NSubB> NChipsPerStaveSB = {NModulesPerStaveSB[IB] * NChipsPerModuleSB[IB],
                                                              NModulesPerStaveSB[MB] * NChipsPerModuleSB[MB],
                                                              NModulesPerStaveSB[OB] * NChipsPerModuleSB[OB]};

  ///< number of cables per stave of sub-barrel
  static constexpr std::array<int, NSubB> NCablesPerStaveSB = {NCablesPerModule[IB] * NModulesPerStaveSB[IB],
                                                               NCablesPerModule[MB] * NModulesPerStaveSB[MB],
                                                               NCablesPerModule[OB] * NModulesPerStaveSB[OB]};

  ///< pattern of cables per stave of sub-barrel
  static constexpr std::array<int, NSubB> CablesOnStaveSB = {(0x1 << NCablesPerModule[IB] * NModulesPerStaveSB[IB]) - 1,
                                                             (0x1 << NCablesPerModule[MB] * NModulesPerStaveSB[MB]) - 1,
                                                             (0x1 << NCablesPerModule[OB] * NModulesPerStaveSB[OB]) - 1};

  ///< number of chips per sub-barrel
  static constexpr std::array<int, NSubB> NChipsSB = {NChipsPerStaveSB[IB] * NStavesSB[IB],
                                                      NChipsPerStaveSB[MB] * NStavesSB[MB],
                                                      NChipsPerStaveSB[OB] * NStavesSB[OB]};

  static constexpr int NChips = NChipsSB[IB] + NChipsSB[MB] + NChipsSB[OB];

  ///< mapping from SW chips ID within the module to HW ID
  /*
    SW/HW correspondence
     13/14|12/13|11/12|10/11| 9/10| 8/ 9| 7/ 8
    ----- ----- ----- ----- ----- ----- -----
     0/ 0| 1/ 1| 2/ 2| 3/ 3| 4/ 4| 5/ 5| 6/ 6
   */
  // SW ID -> HW ID within the module
  static constexpr std::uint8_t ChipOBModSW2HW[14] = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14};
  // HW ID -> SW ID within the module
  static constexpr std::uint8_t ChipOBModHW2SW[15] = {0, 1, 2, 3, 4, 5, 6, 255, 7, 8, 9, 10, 11, 12, 13};

  /// info per stave
  std::array<RUInfo, NStavesSB[IB] + NStavesSB[MB] + NStavesSB[OB]> mStavesInfo;
  std::vector<uint8_t> mFEEId2RUSW; // HW RU ID -> SW ID conversion

  // info on chips info within the stave
  std::array<ChipOnRUInfo, NChipsPerStaveSB[IB] + NChipsPerStaveSB[MB] + NChipsPerStaveSB[OB]> mChipsInfo;
  int mChipInfoEntrySB[NSubB] = {0};

  std::vector<uint8_t> mCableHW2SW[NSubB];       ///< table of cables HW to SW conversion for each RU type
  std::vector<uint8_t> mCableHWFirstChip[NSubB]; ///< 1st chip of module (relative to the 1st chip of the stave) served by each cable

  ClassDefNV(ChipMappingITS, 1);
};
} // namespace itsmft
} // namespace o2

#endif
