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

  o2::Base::GeometryManager::loadGeometry(inputGeom, "FAIRGeom");
  auto gm = o2::MFT::GeometryTGeo::Instance(); // geometry manager for mapping

  // ladder (MFT) = module (ITS)
  Int_t chip, ladder, layer, disk, half, ladderP = -1, layerP = -1, halfP = -1;
  Int_t sensor, nChipsPerLadder;
  Int_t nModules = 0;
  Int_t nChips = gm->getNumberOfChips();
  std::vector<Int_t> modFirstChip, modNChips;

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
      fprintf(srcFile, "// chip: %3d (%1d), ladder: %2d, layer: %1d, disk: %1d, half: %1d\n", iChip, nChipsPerLadder, ladder, layer, disk, half);
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
