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
  FILE* headerFile = fopen("ChipMappingITS.h", "w");

  int lay, sta, ssta, mod, chip, modP = -1, staP = -1, sstaP = -1, layP = -1;
  int nModules = 0;
  int nchTot = gm->getNumberOfChips();
  std::vector<int> modFirstChip, modNChips;
  fprintf(headerFile,
          "// Copyright CERN and copyright holders of ALICE O2. This software is\n"
          "// distributed under the terms of the GNU General Public License v3 (GPL\n"
          "// Version 3), copied verbatim in the file \"COPYING\".\n"
          "//\n"
          "// See http://alice-o2.web.cern.ch/license for full licensing information.\n"
          "//\n"
          "// In applying this license CERN does not waive the privileges and immunities\n"
          "// granted to it by virtue of its status as an Intergovernmental Organization\n"
          "// or submit itself to any jurisdiction.\n\n"
          "#ifndef ALICEO2_ITS_CHIPMAPPING_H\n"
          "#define ALICEO2_ITS_CHIPMAPPING_H\n\n\n"
          "// \\file ChipMapping.h \n"
          "// \\brief Autimatically generated ITS chip <-> module mapping\n\n\n");

  fprintf(headerFile,
          "#include <Rtypes.h>\n"
          "#include <array>\n\n");
  fprintf(headerFile,
          "namespace o2 {\n"
          "namespace its {\n\n");
  fprintf(headerFile,
          "struct ITSChipMappingData {\n"
          "  UShort_t module = 0;       // global module ID\n"
          "  UChar_t  chipInModule = 0; // chip within the module\n"
          "  ClassDefNV(ITSChipMappingData,1);\n"
          "};\n\n"
          "struct ITSModuleMappingData {\n"
          "  UChar_t nChips = 0; // number of chips\n"
          "  UShort_t firstChipID = 0; // global id of 1st chip\n"
          "  ClassDefNV(ITSModuleMappingData,1);\n"
          "};\n\n"
          "class ChipMappingITS {\n"
          " public:\n\n"
          "  constexpr int getNModules() { return NModules; }\n\n"
          "  constexpr int getNChips() { return NChips; }\n\n"
          "  int chipID2Module(int chipID, int &chipInModule) {\n"
          "    chipInModule = mITSChipMappingData[chipID].chipInModule;\n"
          "    return mITSChipMappingData[chipID].module;\n"
          "  }\n\n"
          "  int getNChipsInModule(int modID) {\n"
          "   return mITSModuleMappingData[modID].nChips;\n"
          "  }\n\n"
          "  int module2ChipID(int modID, int chipInModule) {\n"
          "   return mITSModuleMappingData[modID].firstChipID + chipInModule;\n"
          "  }\n\n"
          " private:\n\n");
  fprintf(headerFile,
          "  static constexpr int NChips = %d;\n\n", nchTot);
  fprintf(headerFile,
          "  // moduleGlo chipInMod // chipGlo : lr : stave : hstave : modInStave\n"
          "  std::array<ITSChipMappingData,NChips> mITSChipMappingData = {{\n");
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
      fprintf(headerFile, "  // chip:%d layer: %d, loc.stave: %d (h-stave: %d), loc.module: %d\n",
              ich, lay, sta, ssta, mod);
    }
    //
    fprintf(headerFile, " {%d, %d}%s", nModules - 1, chip, ich < nchTot - 1 ? "," : "");
    int nLn = modNChips.back();
    if (nLn > 10 && chip < nLn / 2) {
      nLn /= 2;
    }
    if (chip + 1 == nLn) {
      fprintf(headerFile, "\n");
    }
  }
  fprintf(headerFile, " }};\n\n");
  fprintf(headerFile, " static constexpr int NModules = %d;\n\n", nModules);
  fprintf(headerFile, " std::array<ITSModuleMappingData,NModules> mITSModuleMappingData = {{\n");
  layP = -1;
  for (int imd = 0; imd < nModules; imd++) {
    gm->getChipId(modFirstChip[imd], lay, sta, ssta, mod, chip);
    if (lay != layP) {
      layP = lay;
      fprintf(headerFile, "  // layer: %d\n", lay);
    }
    fprintf(headerFile, " {%d, %d}%s", modNChips[imd], modFirstChip[imd], imd < nModules - 1 ? "," : "");
    if (imd && (imd % 7) == 0) {
      fprintf(headerFile, "\n");
    }
  }
  fprintf(headerFile, " }};\n");
  fprintf(headerFile,
          " ClassDefNV(ChipMappingITS,1)\n"
          "};\n}\n}\n\n\n");
  fprintf(headerFile, "#endif\n");
  fclose(headerFile);
}
