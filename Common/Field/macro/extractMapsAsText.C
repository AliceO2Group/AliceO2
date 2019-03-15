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

// This macro converts all mag.field objects in the inpFileName file to text files named as <prefix><map_name>
// in current directory

int extractMapsAsText(const std::string inpFileName = "$O2_ROOT/share/Common/maps/mfchebKGI_sym.root",
                      const std::string prefix = "_alice_mag_map_")
{
  int nMaps = 0;
  TFile inpFile(inpFileName.c_str());
  if (!inpFile.IsOpen() || inpFile.IsZombie()) {
    std::cout << "Failed to open file " << inpFileName << " with maps\n";
    return nMaps;
  }

  TIter nextKey(inpFile.GetListOfKeys());
  TObject* obj = nullptr;
  while ((obj = nextKey())) {
    if (obj->IsA() == TKey::Class()) {
      auto magmap = dynamic_cast<MapClass*>(inpFile.Get(obj->GetName()));
      if (!magmap) {
        std::cout << "Failed to load map " << obj->GetName();
        return -nMaps;
      }
      std::string outName = prefix + obj->GetName() + ".txt";
      std::cout << "Dumping map " << obj->GetName() << " as a text to " << outName << "\n";
      magmap->saveData(outName.c_str());
      nMaps++;
    }
  }
  return nMaps;
}
