// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// \file ChipMappingITS.cxx
// \brief Autimatically generated ITS chip <-> module mapping

#include <FairLogger.h>
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include <cassert>
#include <sstream>
#include <iomanip>

using namespace o2::itsmft;

constexpr std::array<int, ChipMappingITS::NSubB> ChipMappingITS::NModulesAlongStaveSB;

constexpr std::array<int, ChipMappingITS::NSubB> ChipMappingITS::NChipsPerModuleSB;
constexpr std::array<int, ChipMappingITS::NSubB> ChipMappingITS::NModulesPerStaveSB;
constexpr std::array<int, ChipMappingITS::NSubB> ChipMappingITS::NCablesPerStaveSB;

constexpr std::array<int, ChipMappingITS::NSubB> ChipMappingITS::NStavesSB;
constexpr std::array<int, ChipMappingITS::NSubB> ChipMappingITS::NChipsPerStaveSB;
constexpr std::array<int, ChipMappingITS::NSubB> ChipMappingITS::NChipsPerCableSB;
constexpr std::array<int, ChipMappingITS::NSubB> ChipMappingITS::NChipsSB;
constexpr std::array<int, ChipMappingITS::NLayers> ChipMappingITS::NStavesOnLr;
constexpr std::array<int, ChipMappingITS::NLayers> ChipMappingITS::FirstStaveOnLr;

constexpr std::array<uint8_t, ChipMappingITS::NLayers> ChipMappingITS::RUTypeLr;

constexpr std::array<uint8_t, ChipMappingITS::NSubB> ChipMappingITS::GBTHeaderFlagSB;
constexpr std::uint8_t ChipMappingITS::ChipOBModSW2HW[14];
constexpr std::uint8_t ChipMappingITS::ChipOBModHW2SW[15];

//______________________________________________
ChipMappingITS::ChipMappingITS()
{
  // init chips info

  /*
    FEE ID field in RDH is 16-bit wide
    Each RU has 10-bit DIPSWITCH, use bits 9:2 as an 8bit ID field to be mapped to FEE ID field in RDH

    | Lr |Stave Count| Bin. pref | Range of Binary Addresses
    |    |= RU Count | Stave addr| 
    |----------------------------------------------------------
    | L0 |   12      |b 0000xxxx | 0000_0000 : 0000_1011 (0 – 11)
    | L1 |   16      |b 0001xxxx | 0001_0000 : 0001_1111 (0 – 15)
    | L2 |   20      |b 001xxxxx | 001_00000 : 001_10011 (0 – 19)
    | L3 |   24      |b 010xxxxx | 010_00000 : 010_10111 (0 – 23)
    | L4 |   30      |b 011xxxxx | 011_00000 : 011_11101 (0 – 29)
    | L5 |   42      |b 10xxxxxx | 10_000000 : 10_101001 (0 – 41)
    | L6 |   48      |b 11xxxxxx | 11_000000 : 11_101111 (0 – 47)

    FEEId format:
    15|14   12|11    10 |9     8|7  6|5           0|
     0| Layer | Reserve | Fiber | 00 | StaveNumber |

   */

  uint32_t maxRUHW = composeFEEId(NLayers - 1, NStavesOnLr[NLayers - 1], NLinks - 1); // Max possible FEE ID
  assert(maxRUHW < 0xffff);
  mFEEId2RUSW.resize(maxRUHW + 1, 0xff);

  // IB: single cable per chip
  int ctrChip = 0;
  mChipInfoEntrySB[IB] = ctrChip;
  mCablePos[IB].resize(NChipsPerStaveSB[IB], 0xff);
  mCableHW2SW[IB].resize(NChipsPerStaveSB[IB], 0xff);
  mCableHW2Pos[IB].resize(NChipsPerStaveSB[IB], 0xff);
  mCableHWFirstChip[IB].resize(NChipsPerStaveSB[IB], 0xff);
  for (int i = 0; i < NChipsPerStaveSB[IB]; i++) {
    auto& cInfo = mChipsInfo[ctrChip++];
    cInfo.id = i;
    cInfo.moduleHW = 0;
    cInfo.moduleSW = 0;
    cInfo.chipOnModuleSW = i;
    cInfo.chipOnModuleHW = i;
    cInfo.cableHW = i;                              //1-to-1 mapping
    cInfo.cableHWPos = i;                           //1-to-1 mapping
    cInfo.cableSW = i;                              //1-to-1 mapping
    cInfo.chipOnCable = 0;                          // every chip is master
    mCableHW2SW[IB][cInfo.cableHW] = cInfo.cableSW;
    mCableHW2Pos[IB][cInfo.cableHW] = cInfo.cableHWPos;
    mCablesOnStaveSB[IB] |= 0x1 << cInfo.cableHWPos; // account in lanes pattern
    mCableHWFirstChip[IB][i] = 0;                   // stave and module are the same
  }

  // [i][j] gives lane id for  lowest(i=0) and highest(i=1) 7 chips of HW module (j+1) (1-4 for ML, 1-7 for OL)
  const int LANEID[2][7] = {{6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6}};
  const int maxModulesPerStave = NModulesPerStaveSB[OB];
  const int chipsOnCable = 7;
  for (int bid = MB; bid <= OB; bid++) { // MB and OB staves have similar layout
    mChipInfoEntrySB[bid] = ctrChip;
    mCablePos[bid].resize(NChipsPerStaveSB[bid], 0xff);
    mCableHW2SW[bid].resize(NChipsPerStaveSB[bid], 0xff);
    mCableHW2Pos[bid].resize(NChipsPerStaveSB[bid], 0xff);
    mCableHWFirstChip[bid].resize(NChipsPerStaveSB[bid], 0xff);
    for (int i = 0; i < NChipsPerStaveSB[bid]; i++) {
      auto& cInfo = mChipsInfo[ctrChip++];
      int hstave = i / (NChipsPerStaveSB[bid] / 2);
      cInfo.id = i;
      cInfo.moduleSW = i / NChipsPerModuleSB[bid];                         // SW module ID (within the stave)
      cInfo.moduleHW = 1 + cInfo.moduleSW % (NModulesPerStaveSB[bid] / 2); // identification within the HS

      cInfo.chipOnModuleSW = i % NChipsPerModuleSB[bid];
      cInfo.chipOnModuleHW = ChipOBModSW2HW[cInfo.chipOnModuleSW];

      bool upper7 = cInfo.chipOnModuleSW >= (NChipsPerModuleSB[bid] / 2);

      uint8_t connector = 2 * hstave + upper7;
      cInfo.cableHW = (connector << 3) + LANEID[upper7][cInfo.moduleHW - 1];
      cInfo.cableSW = i / chipsOnCable;
      cInfo.cableHWPos = LANEID[upper7][cInfo.moduleHW - 1] + connector * maxModulesPerStave / 2;
      mCablesOnStaveSB[bid] |= 0x1 << cInfo.cableHWPos;                        // account in lanes pattern
      cInfo.chipOnCable = cInfo.chipOnModuleSW % (NChipsPerModuleSB[bid] / 2); // each cable serves half module
      mCableHW2SW[bid][cInfo.cableHW] = cInfo.cableSW;
      mCableHW2Pos[bid][cInfo.cableHW] = cInfo.cableHWPos;
      if (cInfo.chipOnCable == 0) {
        mCableHWFirstChip[bid][cInfo.cableHW] = cInfo.moduleSW * NChipsPerModuleSB[bid];
      }
    }
  }
  for (int bid = 0; bid < NSubB; bid++) {
    int pos = 0;
    for (int ic = 0; ic < 32; ic++) {
      if (mCablesOnStaveSB[bid] & (0x1 << ic)) {
        mCablePos[bid][pos++] = ic;
      }
    }
    if (pos != NCablesPerStaveSB[bid]) {
      throw std::runtime_error(fmt::format("counted number of cables {} does not match expected {} on subBarel{}", pos, NCablesPerStaveSB[bid], bid));
    }
  }

  int ctrStv = 0;
  uint16_t chipCount = 0;
  for (int ilr = 0; ilr < NLayers; ilr++) {
    for (int ist = 0; ist < NStavesOnLr[ilr]; ist++) {
      auto& sInfo = mStavesInfo[ctrStv];
      sInfo.idSW = ctrStv++;

      // map FEEIds (RU read out by at most 3 GBT links) to SW ID
      sInfo.idHW = composeFEEId(ilr, ist, 0); // FEEId for link 0
      mFEEId2RUSW[sInfo.idHW] = sInfo.idSW;
      for (int lnk = 1; lnk < NLinks; lnk++) {
        mFEEId2RUSW[composeFEEId(ilr, ist, lnk)] = sInfo.idSW;
      }
      sInfo.layer = ilr;
      sInfo.ruType = RUTypeLr[ilr];
      sInfo.nCables = NCablesPerStaveSB[sInfo.ruType];
      sInfo.firstChipIDSW = chipCount;
      chipCount += NChipsPerStaveSB[sInfo.ruType];
    }
  }
  assert(ctrStv == getNRUs());
}

//______________________________________________
void ChipMappingITS::print() const
{
  int ctrChip = 0;
  printf("\n\nSubBarrel IB\n");
  for (int i = 0; i < NChipsPerStaveSB[IB]; i++) {
    mChipsInfo[ctrChip++].print();
  }
  printf("\n\nSubBarrel MB\n");
  for (int i = 0; i < NChipsPerStaveSB[MB]; i++) {
    mChipsInfo[ctrChip++].print();
  }
  printf("\n\nSubBarrel OB\n");
  for (int i = 0; i < NChipsPerStaveSB[OB]; i++) {
    mChipsInfo[ctrChip++].print();
  }
}

//______________________________________________
void ChipMappingITS::expandChipInfoSW(int idSW, int& lay, int& sta, int& ssta, int& mod, int& chipInMod) const
{
  // convert SW chip ID to detailed info SW info
  ChipInfo chi;
  getChipInfoSW(idSW, chi);
  const auto staveInfo = getRUInfoSW(chi.ru);
  lay = staveInfo->layer;
  sta = staveInfo->idSW;
  mod = chi.chOnRU->moduleSW;
  chipInMod = chi.chOnRU->chipOnModuleSW;
  ssta = lay < 3 || (mod < (NModulesPerStaveSB[chi.ruType] >> 2)) ? 0 : 1;
}

//______________________________________________
void ChipMappingITS::expandChipInfoHW(int idSW, int& lay, int& sta, int& ssta, int& mod, int& chipInMod) const
{
  // convert SW chip ID to detailed info HW info
  ChipInfo chi;
  getChipInfoSW(idSW, chi);
  const auto staveInfo = getRUInfoSW(chi.ru);
  lay = staveInfo->layer;
  sta = staveInfo->idSW - getFirstStavesOnLr(lay);                                   // stave relative to layer
  mod = lay < 3 ? 0 : 1 + (chi.chOnRU->moduleSW % NModulesAlongStaveSB[chi.ruType]); // module on the substave
  ssta = chi.chOnRU->moduleSW / NModulesAlongStaveSB[chi.ruType];
  chipInMod = chi.chOnRU->chipOnModuleHW;
}

//______________________________________________
std::string ChipMappingITS::getChipNameHW(int idSW) const
{
  // convert global SW chip ID to name in HW conventions
  int lay, sta, ssta, mod, cinmod;
  expandChipInfoHW(idSW, lay, sta, ssta, mod, cinmod);
  std::stringstream strs;
  strs << 'L' << lay << '_' << std::setfill('0') << std::setw(2) << sta;
  if (lay > 2) {
    strs << (ssta ? 'U' : 'L') << "_M" << mod;
  }
  strs << "_C" << std::setfill('0') << std::setw(2) << cinmod;

  return strs.str();
}

///< impose user defined FEEId -> ruSW (staveID) conversion, to be used only for forced decoding of corrupted data
void ChipMappingITS::imposeFEEId2RUSW(uint16_t feeID, uint16_t ruSW)
{
  // test if it is legitimate
  uint16_t lr, ruOnLr, link;
  expandFEEId(feeID, lr, ruOnLr, link);
  if (lr >= NLayers || ruOnLr >= NStavesOnLr[lr] || link >= NLinks) {
    LOG(FATAL) << "Invalid FEE#0x" << std::hex << feeID << std::dec << ": corresponds to Lr#" << lr
               << " StaveOnLr#" << ruOnLr << " GBTLinkOnRU#" << link;
  }
  if (ruSW >= getNRUs()) {
    LOG(FATAL) << "Invalid SW RUid " << ruSW << " (cannot exceed " << getNRUs() << ")";
  }
  mFEEId2RUSW[feeID] = ruSW;
}
