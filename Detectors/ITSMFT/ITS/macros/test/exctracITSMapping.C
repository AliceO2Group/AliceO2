#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <FairLogger.h>
#include <stdio.h>
#include <vector>
#include "ITSBase/GeometryTGeo.h"
#endif

/*
  This macro generates the mapping 
  global chipID  <-> {global module ID, chip in module}
  needed for the raw data pixel reader
  The mapping is generated from the local geometry file and
  should be put in the directory on which the PixelReader may depend
*/
void extractITSMapping(const std::string inputGeom = "O2geometry.root")
{
  o2::Base::GeometryManager::loadGeometry(inputGeom, "FAIRGeom");
  auto gm = o2::ITS::GeometryTGeo::Instance(); // geometry manager for mapping
  int nchips = gm->getNumberOfChips();
  FILE* srcFile = fopen("ChipMappingITS.cxx", "w");

  int lay, sta, ssta, mod, chip, modP = -1, staP = -1, sstaP = -1, layP = -1;
  int nModules = 0;
  int nchTot = gm->getNumberOfChips();
  std::vector<int> modFirstChip, modNChips;
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
          "// \\brief Autimatically generated ITS chip <-> module mapping\n\n\n");

  fprintf(srcFile, "%s\n\n", R"(#include "ITSMFTReconstruction/ChipMappingITS.h")");
  fprintf(srcFile, "using namespace o2::ITSMFT;\n\n\n");
  fprintf(srcFile,
          "// moduleGlo chipInMod // chipGlo : lr : stave : hstave : modInStave\n"
          "const std::array<ITSChipMappingData,ChipMappingITS::NChips> ChipMappingITS::ChipMappingData\n{{\n");
  for (int ich = 0; ich < nchTot; ich++) {
    gm->getChipId(ich, lay, sta, ssta, mod, chip);
    if (lay != layP || mod != modP || sta != staP || ssta != sstaP) { // new module
      modP = mod;
      staP = sta;
      sstaP = ssta;
      layP = lay;
      modFirstChip.push_back(ich);
      modNChips.push_back(gm->getNumberOfChipsPerModule(lay));
      nModules++;
      fprintf(srcFile, "// chip:%d layer: %d, loc.stave: %d (h-stave: %d), loc.module: %d\n",
              ich, lay, sta, ssta, mod);
    }
    //
    fprintf(srcFile, " {%d, %d}%s", nModules - 1, chip, ich < nchTot - 1 ? "," : "");
    int nLn = modNChips.back();
    if (chip + 1 == nLn) {
      fprintf(srcFile, "\n");
    }
  }
  fprintf(srcFile, "\n}};\n\n");
  fprintf(srcFile,
          "const std::array<ITSModuleMappingData,ChipMappingITS::NModules> "
          "ChipMappingITS::ModuleMappingData\n{{\n");
  layP = -1;
  for (int imd = 0; imd < nModules; imd++) {
    gm->getChipId(modFirstChip[imd], lay, sta, ssta, mod, chip);
    if (lay != layP) {
      layP = lay;
      fprintf(srcFile, "\n// layer: %d\n", lay);
    }
    fprintf(srcFile, " {%d, %d, %d}%s", lay, modNChips[imd], modFirstChip[imd], imd < nModules - 1 ? "," : "");
    if (imd && (imd % 7) == 0) {
      fprintf(srcFile, "\n");
    }
  }
  fprintf(srcFile, "\n}};\n");
  fclose(srcFile);
}
