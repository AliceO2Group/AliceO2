#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <FairLogger.h>
#include <stdio.h>
#include <vector>
#include "MFTBase/GeometryTGeo.h"
#endif
/*
  This macro generates the mapping 
  global chipID  <-> {global module ID, chip in module}
  needed for the raw data pixel reader
  The mapping is generated from the local geometry file and
  should be put in the directory on which the PixelReader may depend
*/
void extractMFTMapping(const std::string inputGeom = "O2geometry.root")
{

  o2::base::GeometryManager::loadGeometry(inputGeom, "FAIRGeom");
  auto gm = o2::MFT::GeometryTGeo::Instance(); // geometry manager for mapping

  // ladder (MFT) = module (ITS)
  Int_t chip, ladder, layer, disk, half, ladderP = -1, layerP = -1, halfP = -1;
  Int_t ladderID, zone;
  Int_t sensor, nChipsPerLadder;
  Int_t nModules = 0;
  Int_t nChips = gm->getNumberOfChips();
  std::vector<Int_t> modFirstChip, modNChips;

  FILE* srcFile = fopen("ChipMappingMFT_zones.cxx", "w");

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
  fprintf(srcFile, "using namespace o2::ITSMFT;\n");
  fprintf(srcFile, "const std::array<MFTChipMappingData,ChipMappingMFT::NChips> ChipMappingMFT::ChipMappingData\n{{\n");

  for (Int_t iChip = 0; iChip < nChips; iChip++) {
    gm->getSensorID(iChip, half, disk, ladder, sensor);
    layer = gm->getLayer(iChip);
    //printf("%4d   %1d   %2d   %1d   %1d   %1d \n",iChip,sensor,ladder,layer,disk,half);
    if (layer != layerP || ladder != ladderP || half != halfP) { // new module
      layerP = layer;
      ladderP = ladder;
      halfP = half;
      modFirstChip.push_back(iChip);
      nChipsPerLadder = gm->getNumberOfSensorsPerLadder(half, disk, ladder);
      modNChips.push_back(nChipsPerLadder);
      nModules++;
      //fprintf(srcFile, "// chip: %3d (%1d), ladder: %2d, layer: %1d, disk: %1d, half: %1d\n", iChip, nChipsPerLadder, ladder, layer, disk, half);
      ladderID = gm->getLadderID(disk, ladder);
      if (disk == 0) {
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
      }
      if (disk == 1) {
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
      }
      if (disk == 2) {
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
      }
      if (disk == 3) {
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
      }
      if (disk == 4) {
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
      }
      fprintf(srcFile, "// chip: %3d (%1d), ladder: %2d (%2d), layer: %1d, disk: %1d, half: %1d, zone: %1d \n", iChip, nChipsPerLadder, ladder, ladderID, layer, disk, half, zone);
    }
    fprintf(srcFile, "{%d, %d}%s\n", (nModules - 1), sensor, (iChip < nChips - 1 ? "," : ""));
  }

  fprintf(srcFile, "\n}};\n\n");

  fprintf(srcFile,
          "const std::array<MFTModuleMappingData,ChipMappingMFT::NModules> "
          "ChipMappingMFT::ModuleMappingData\n{{\n");

  layerP = -1;
  for (int iMod = 0; iMod < nModules; iMod++) {
    gm->getSensorID(modFirstChip[iMod], half, disk, ladder, sensor);
    layer = gm->getLayer(modFirstChip[iMod]);
    if (layer != layerP) {
      layerP = layer;
      fprintf(srcFile, "\n// layer: %d\n", layer);
    }
    fprintf(srcFile, " {%d, %d, %d}%s\n", layer, modNChips[iMod], modFirstChip[iMod], (iMod < nModules - 1 ? "," : ""));
    /*
    if (iMod && (iMod % 7) == 0) {
      fprintf(srcFile, "\n");
    }
    */
  }

  fprintf(srcFile, "\n}};\n");
  fclose(srcFile);
}
