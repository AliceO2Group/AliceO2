#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "Field/MagneticWrapperChebyshev.h"
#include <TSystem.h>
#include <TFile.h>
#include <TKey.h>
#include <TList.h>
#include <TIterator.h>
#include <string>
#include <iostream>
#endif

using MapClass = o2::field::MagneticWrapperChebyshev;

// This macro converts all text files with name pattern <path>/<prefix>* to
// mag.field object of MapClass and stores them in the outFileName root file

int createMapsFromText(const std::string path = "./", // where to look for the text files
                       const std::string outFileName = "mfchebKGI_sym.root",
                       const std::string prefix = "_alice_mag_map_")
{
  int nMaps = 0;
  TFile outFile(outFileName.c_str(), "recreate");
  if (!outFile.IsOpen() || outFile.IsZombie()) {
    std::cout << "Failed to open file " << outFileName << " to store the maps\n";
    return nMaps;
  }
  auto dir = gSystem->OpenDirectory(path.c_str());
  if (!dir) {
    std::cout << "Failed to check the path " << path << "\n";
  }
  const char* np = nullptr;
  while ((np = gSystem->GetDirEntry(dir))) {
    std::string inpName = np;

    if (inpName.rfind(prefix, 0) == 0) { // this is a mag. map text file
      std::cout << "Creating map from " << inpName << "\n";
      MapClass magmap;
      magmap.loadData(inpName.c_str());
      std::cout << "Saving new map " << magmap.GetName() << "\n";
      magmap.Write();
      nMaps++;
    }
  }
  outFile.Close();
  gSystem->FreeDirectory(dir);

  return nMaps;
}
