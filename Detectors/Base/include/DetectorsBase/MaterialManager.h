// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Detector.h
/// \brief Definition of the MaterialManager class

#ifndef ALICEO2_BASE_MATERIALMANAGER_H_
#define ALICEO2_BASE_MATERIALMANAGER_H_

#include "Rtypes.h"
#include <map>

namespace o2
{
namespace Base
{
// Central class managing creation of material, mixtures, mediums through
// the VMC interface. All material creations should go through this class.
// Also manages material indices and offers ways to retrieve them
// given a name (such as a given module name).
class MaterialManager
{
 public:
  static MaterialManager& Instance()
  {
    static MaterialManager inst;
    return inst;
  }

  // Module composition
  void Material(const char* modname, Int_t imat, const char* name, Float_t a, Float_t z, Float_t dens, Float_t radl,
                Float_t absl, Float_t* buf = nullptr, Int_t nwbuf = 0);

  void Mixture(const char* modname, Int_t imat, const char* name, Float_t* a, Float_t* z, Float_t dens, Int_t nlmat,
               Float_t* wmat);

  void Medium(const char* modname, Int_t numed, const char* name, Int_t nmat, Int_t isvol, Int_t ifield, Float_t fieldm,
              Float_t tmaxfd, Float_t stemax, Float_t deemax, Float_t epsil, Float_t stmin, Float_t* ubuf = nullptr,
              Int_t nbuf = 0);

  // insert material name
  void insertMaterialName(const char* uniquename, int index);
  void insertMediumName(const char* uniquename, int index);

  // returns global material ID given a "local" material ID for this detector
  // returns -1 in case local ID not found
  int getMaterialID(const char* modname, int imat) const
  {
    auto lookupiter = mMaterialMap.find(modname);
    if (lookupiter == mMaterialMap.end()) {
      return -1;
    }
    auto lookup = lookupiter->second;

    auto iter = lookup.find(imat);
    if (iter != lookup.end()) {
      return iter->second;
    }
    return -1;
  }

  // returns global medium ID given a "local" medium ID for this detector
  // returns -1 in case local ID not found
  int getMediumID(const char* modname, int imed) const
  {
    auto lookupiter = mMediumMap.find(modname);
    if (lookupiter == mMediumMap.end()) {
      return -1;
    }
    auto lookup = lookupiter->second;

    auto iter = lookup.find(imed);
    if (iter != lookup.end()) {
      return iter->second;
    }
    return -1;
  }

  // fill the medium index mapping into a standard vector
  // the vector gets sized properly and will be overridden
  void getMediumIDMappingAsVector(const char* modname, std::vector<int>& mapping)
  {
    mapping.clear();

    auto lookupiter = mMediumMap.find(modname);
    if (lookupiter == mMediumMap.end()) {
      return;
    }
    auto lookup = lookupiter->second;

    // get the biggest mapped value (maps are sorted in keys)
    auto maxkey = lookup.rbegin()->first;
    // resize mapping and initialize with -1 by default
    mapping.resize(maxkey + 1, -1);
    // fill vector with entries from map
    for (auto& p : lookup) {
      mapping[p.first] = p.second;
    }
  }

  // print out all registered materials
  void printMaterials() const;

  // print out all registered media
  void printMedia() const;

 private:
  MaterialManager() = default;

  // lookup structures
  std::map<std::string, std::map<int, int>>
    mMaterialMap; // map of name -> map of local index to global index for Materials
  std::map<std::string, std::map<int, int>> mMediumMap; // map of name -> map of local index to global index for Media

  std::map<std::string, int> mMaterialNameToGlobalIndexMap; // map of unique material name to global index
  std::map<std::string, int> mMediumNameToGlobalIndexMap;

  Float_t mDensityFactor = 1.; //! factor that is multiplied to all material densities (ONLY for
  // systematic studies)

  ClassDefNV(MaterialManager, 0);
};
}
}

#endif
