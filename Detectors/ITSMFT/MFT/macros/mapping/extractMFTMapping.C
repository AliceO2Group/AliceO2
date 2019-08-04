#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "MFTBase/GeometryTGeo.h"

constexpr int NChips = 936;
constexpr int NModules = 280;
constexpr int NLayers = 10;
constexpr int NZonesPerLayer = 2 * 4;
constexpr int NConnectors = 5;
constexpr int NMaxChipsPerLadder = 5;
constexpr int NRUTypes = 12;

struct MFTChipMappingData {
  UShort_t module = 0;      // global module ID
  UChar_t chipOnModule = 0; // chip within the module
  UChar_t cable = 0;        // cable in the connector
  UChar_t chipOnRU = 0;     // chip within the RU (SW)
};

struct MFTModuleMappingData {
  UChar_t layer = 0;        // layer id
  UChar_t nChips = 0;       // number of chips
  UShort_t firstChipID = 0; // global id of 1st chip
  UChar_t connector = 0;    // cable connector in a zone
  UChar_t zone = 0;         // read-out zone id
  UChar_t disk = 0;         // disk id
  UChar_t half = 0;         // half id
};

std::array<MFTChipMappingData, NChips> ChipMappingData;
std::array<MFTModuleMappingData, NModules> ModuleMappingData;

Int_t ChipOnRUSW[NRUTypes][NConnectors][NMaxChipsPerLadder];

constexpr Int_t ZoneLadderIDmin[NZonesPerLayer / 2][NLayers]{
  {0, 21, 0, 21, 0, 23, 0, 28, 0, 29},
  {3, 18, 3, 18, 3, 20, 4, 24, 5, 25},
  {6, 15, 6, 15, 6, 17, 8, 20, 9, 21},
  {9, 12, 9, 12, 9, 13, 12, 16, 13, 17}};

constexpr Int_t ZoneLadderIDmax[NZonesPerLayer / 2][NLayers]{
  {2, 23, 2, 23, 2, 25, 3, 31, 4, 33},
  {5, 20, 5, 20, 5, 22, 7, 27, 8, 28},
  {8, 17, 8, 17, 8, 19, 11, 23, 12, 24},
  {11, 14, 11, 14, 12, 16, 15, 19, 16, 20}};

constexpr Int_t ChipConnectorCable[NConnectors][NMaxChipsPerLadder]{
  {5, 6, 7, 24, 23},
  {0, 1, 2, 3, 4},
  {17, 16, 15, 14, 13},
  {22, 21, 20, 19, 18},
  {12, 11, 10, 9, 8}};

constexpr Int_t ZoneRUType[NZonesPerLayer / 2][NLayers / 2]{
  {1, 1, 1, 7, 11},
  {2, 2, 4, 8, 9},
  {2, 2, 3, 8, 10},
  {0, 0, 5, 6, 7}};

///< number of chips per zone (RU)
constexpr std::array<int, NRUTypes> NChipsOnRUType{7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19};

constexpr Int_t RUType[20]{-1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10, 11};

void createCXXfile(o2::mft::GeometryTGeo*);
Int_t getZone(Int_t layer, Int_t ladderID, Int_t& connector);
Int_t getZoneRUID(Int_t half, Int_t layer, Int_t zone, Int_t& zoneID);

Int_t chip, ladder, layer, disk, half, ladderP, layerP, halfP;
Int_t ladderID, zone, connector, iconnector;
Int_t sensor, nChipsPerLadder;
Int_t ruHW, zoneID;
Int_t nModules;
std::vector<Int_t> modFirstChip, modNChips, RUNChips;

void extractMFTMapping(const std::string inputGeom = "O2geometry.root")
{
  o2::base::GeometryManager::loadGeometry(inputGeom, "FAIRGeom");
  auto gm = o2::mft::GeometryTGeo::Instance(); // geometry manager for mapping

  createCXXfile(gm);

  // chip on RU SW
  Int_t chipsOnRUType[NRUTypes]{0};
  Int_t ctrChip = 0;
  for (Int_t iRU = 0; iRU < NRUTypes; ++iRU) {
    for (Int_t layer = 0; layer < NLayers; ++layer) {
      for (Int_t zone = 0; zone < NZonesPerLayer; ++zone) {
        auto ruType = ZoneRUType[zone % 4][layer / 2];
        if (ruType != iRU || chipsOnRUType[iRU] == NChipsOnRUType[iRU]) {
          continue;
        }
        for (Int_t iMod = 0; iMod < NModules; ++iMod) {
          if (ModuleMappingData[iMod].layer != layer)
            continue;
          if ((ModuleMappingData[iMod].zone + (NZonesPerLayer / 2) * ModuleMappingData[iMod].half) != zone)
            continue;
          for (Int_t iChip = 0; iChip < ModuleMappingData[iMod].nChips; ++iChip) {
            Int_t chipID = iChip + ModuleMappingData[iMod].firstChipID;
            // reverse connectors for zone 3 on disk 4 (layers 8 and 9), an RU 7
            iconnector = ModuleMappingData[iMod].connector;
            if ((layer / 2) == 4 && (zone % 4) == 3) {
              iconnector = 3 - ModuleMappingData[iMod].connector;
            }
            ChipOnRUSW[iRU][iconnector][iChip] = chipsOnRUType[iRU];
            ++chipsOnRUType[iRU];
          } // loop over chips per module
          if (chipsOnRUType[iRU] == NChipsOnRUType[iRU]) {
            break;
          }
        } // loop over modules
      }   // loop over zones per layer
    }     // loop over layers
  }       // loop over RU

  // create again the CXX file with the chipOnRUSW member
  createCXXfile(gm);
}

//__________________________________________________________________________
void createCXXfile(o2::mft::GeometryTGeo* gm)
{
  // ladder (MFT) = module (ITS)
  Int_t nChips = gm->getNumberOfChips();
  if (nChips != NChips) {
    printf("Wrong number of chips from the geometry! %d != %d \n", nChips, NChips);
    return;
  }

  nModules = 0;
  ladderP = layerP = halfP = -1;
  for (Int_t i = 0; i < 80; ++i) {
    RUNChips.push_back(0);
  }

  FILE* srcFile = fopen("ChipMappingMFT.cxx", "w");

  fprintf(srcFile,
          "// Copyright CERN and copyright holders of ALICE O2. This software is\n"
          "// distributed under the terms of the GNU General Public License v3 (GPL\n"
          "// Version 3), copied verbatim in the file \"COPYING\".\n"
          "//\n"
          "// See http://alice-o2.web.cern.ch/license for full licensing information.\n"
          "//\n"
          "// In applying this license CERN does not waive the privileges and immunities\n"
          "// granted to it by virtue of its status as an Intergovernmental Organization\n"
          "// or submit itself to any jurisdiction.\n\n"
          "// \\file ChipMappingITS.cxx \n"
          "// \\brief Automatically generated MFT chip <-> module mapping\n");

  fprintf(srcFile, "%s\n\n", R"(#include "ITSMFTReconstruction/ChipMappingMFT.h")");
  fprintf(srcFile, "using namespace o2::itsmft;\n");
  fprintf(srcFile, "const std::array<MFTChipMappingData,ChipMappingMFT::NChips> ChipMappingMFT::ChipMappingData\n{{\n");

  fprintf(srcFile, "\n// { module, chipOnModule, cable, chipOnRU }\n\n");
  for (Int_t iChip = 0; iChip < nChips; iChip++) {
    gm->getSensorID(iChip, half, disk, ladder, sensor);
    layer = gm->getLayer(iChip);
    if (layer != layerP || ladder != ladderP || half != halfP) { // new module
      layerP = layer;
      ladderP = ladder;
      halfP = half;
      modFirstChip.push_back(iChip);
      nChipsPerLadder = gm->getNumberOfSensorsPerLadder(half, disk, ladder);
      modNChips.push_back(nChipsPerLadder);
      nModules++;
      ladderID = gm->getLadderID(disk, ladder);
      zone = getZone(layer, ladderID, connector);
      if (zone < 0) {
        printf("Wrong zone for disk %d layer %d ladderID %d \n", disk, layer, ladderID);
        return;
      }
      fprintf(srcFile, "// chip: %3d (%1d), ladder: %2d (%2d), layer: %1d, disk: %1d, half: %1d, zone: %1d \n", iChip, nChipsPerLadder, ladder, ladderID, layer, disk, half, zone);
    }
    Int_t iconnector = connector;
    if ((layer / 2) == 4 && zone == 3) {
      iconnector = 3 - iconnector;
    }
    fprintf(srcFile, "{%d, %d, %d, %d}%s\n", (nModules - 1), sensor, ChipConnectorCable[connector][sensor], ChipOnRUSW[ZoneRUType[zone][layer / 2]][iconnector][sensor], (iChip < nChips - 1 ? "," : ""));
    ChipMappingData[iChip].module = nModules - 1;
    ChipMappingData[iChip].chipOnModule = sensor;
    ChipMappingData[iChip].cable = ChipConnectorCable[connector][sensor];
  }

  fprintf(srcFile, "\n}};\n\n");

  if (nModules != NModules) {
    printf("Wrong number of modules from the geometry! %d != %d \n", nModules, NModules);
    return;
  }

  fprintf(srcFile,
          "const std::array<MFTModuleMappingData,ChipMappingMFT::NModules> "
          "ChipMappingMFT::ModuleMappingData\n{{\n");

  layerP = -1;
  fprintf(srcFile, "\n// { layer, nChips, firstChipID, connector, zone, disk, half }\n\n");
  for (Int_t iMod = 0; iMod < nModules; iMod++) {
    gm->getSensorID(modFirstChip[iMod], half, disk, ladder, sensor);
    layer = gm->getLayer(modFirstChip[iMod]);
    if (layer != layerP) { // new layer
      layerP = layer;
      fprintf(srcFile, "\n// layer: %d\n", layer);
    }
    ladderID = gm->getLadderID(disk, ladder);
    zone = getZone(layer, ladderID, connector);
    ruHW = getZoneRUID(half, layer, zone, zoneID);
    RUNChips.at(zoneID) += modNChips[iMod];
    if (zone < 0) {
      printf("Wrong zone for disk %d layer %d ladderID %d \n", disk, layer, ladderID);
      return;
    }
    fprintf(srcFile, " {%d, %d, %d, %d, %d, %d, %d}%s\n", layer, modNChips[iMod], modFirstChip[iMod], connector, zone, disk, half, (iMod < nModules - 1 ? "," : ""));
    ModuleMappingData[iMod].layer = layer;
    ModuleMappingData[iMod].nChips = modNChips[iMod];
    ModuleMappingData[iMod].firstChipID = modFirstChip[iMod];
    ModuleMappingData[iMod].connector = connector;
    ModuleMappingData[iMod].zone = zone;
    ModuleMappingData[iMod].disk = disk;
    ModuleMappingData[iMod].half = half;
  }

  fprintf(srcFile, "\n}};\n");
  fclose(srcFile);
}

//__________________________________________________________________________
Int_t getZone(Int_t layer, Int_t ladderID, Int_t& connector)
{
  Int_t zone = -1;
  if (layer == 0) {
    if (ladderID >= 0 && ladderID <= 2)
      zone = 0;
    if (ladderID >= 3 && ladderID <= 5)
      zone = 1;
    if (ladderID >= 6 && ladderID <= 8)
      zone = 2;
    if (ladderID >= 9 && ladderID <= 11)
      zone = 3;
  }
  if (layer == 1) {
    if (ladderID >= 12 && ladderID <= 14)
      zone = 3;
    if (ladderID >= 15 && ladderID <= 17)
      zone = 2;
    if (ladderID >= 18 && ladderID <= 20)
      zone = 1;
    if (ladderID >= 21 && ladderID <= 23)
      zone = 0;
  }
  if (layer == 2) {
    if (ladderID >= 0 && ladderID <= 2)
      zone = 0;
    if (ladderID >= 3 && ladderID <= 5)
      zone = 1;
    if (ladderID >= 6 && ladderID <= 8)
      zone = 2;
    if (ladderID >= 9 && ladderID <= 11)
      zone = 3;
  }
  if (layer == 3) {
    if (ladderID >= 12 && ladderID <= 14)
      zone = 3;
    if (ladderID >= 15 && ladderID <= 17)
      zone = 2;
    if (ladderID >= 18 && ladderID <= 20)
      zone = 1;
    if (ladderID >= 21 && ladderID <= 23)
      zone = 0;
  }
  if (layer == 4) {
    if (ladderID >= 0 && ladderID <= 2)
      zone = 0;
    if (ladderID >= 3 && ladderID <= 5)
      zone = 1;
    if (ladderID >= 6 && ladderID <= 8)
      zone = 2;
    if (ladderID >= 9 && ladderID <= 12)
      zone = 3;
  }
  if (layer == 5) {
    if (ladderID >= 13 && ladderID <= 16)
      zone = 3;
    if (ladderID >= 17 && ladderID <= 19)
      zone = 2;
    if (ladderID >= 20 && ladderID <= 22)
      zone = 1;
    if (ladderID >= 23 && ladderID <= 25)
      zone = 0;
  }
  if (layer == 6) {
    if (ladderID >= 0 && ladderID <= 3)
      zone = 0;
    if (ladderID >= 4 && ladderID <= 7)
      zone = 1;
    if (ladderID >= 8 && ladderID <= 11)
      zone = 2;
    if (ladderID >= 12 && ladderID <= 15)
      zone = 3;
  }
  if (layer == 7) {
    if (ladderID >= 16 && ladderID <= 19)
      zone = 3;
    if (ladderID >= 20 && ladderID <= 23)
      zone = 2;
    if (ladderID >= 24 && ladderID <= 27)
      zone = 1;
    if (ladderID >= 28 && ladderID <= 31)
      zone = 0;
  }
  if (layer == 8) {
    if (ladderID >= 0 && ladderID <= 4)
      zone = 0;
    if (ladderID >= 5 && ladderID <= 8)
      zone = 1;
    if (ladderID >= 9 && ladderID <= 12)
      zone = 2;
    if (ladderID >= 13 && ladderID <= 16)
      zone = 3;
  }
  if (layer == 9) {
    if (ladderID >= 17 && ladderID <= 20)
      zone = 3;
    if (ladderID >= 21 && ladderID <= 24)
      zone = 2;
    if (ladderID >= 25 && ladderID <= 28)
      zone = 1;
    if (ladderID >= 29 && ladderID <= 33)
      zone = 0;
  }

  Int_t z;
  for (z = 0; z < (NZonesPerLayer / 2); ++z) {
    if (ladderID >= ZoneLadderIDmin[z][layer] && ladderID <= ZoneLadderIDmax[z][layer])
      break;
  }
  if (zone != z)
    printf("different zone: %d %d \n", zone, z);

  connector = (layer % 2 == 0) ? (ladderID - ZoneLadderIDmin[z][layer]) : (ZoneLadderIDmax[z][layer] - ladderID);

  return zone;
}

//__________________________________________________________________________
Int_t getZoneRUID(Int_t half, Int_t layer, Int_t zone, Int_t& zoneID)
{
  // counting RU SW per half layers
  //zoneID = NLayers * (NZonesPerLayer / 2) * half + (NZonesPerLayer / 2) * layer + zone;
  // counting RU SW per full layers
  zoneID = NZonesPerLayer * layer + (NZonesPerLayer / 2) * half + zone;

  Int_t zoneRUID = 0;
  zoneRUID = 0;
  zoneRUID += half << 6;
  zoneRUID += (layer / 2) << 3; // disk
  zoneRUID += (layer % 2) << 2; // plane (disk face)
  zoneRUID += zone;

  return zoneRUID;
}

#endif
