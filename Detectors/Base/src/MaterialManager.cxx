// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MaterialManager.cxx
/// \brief Implementation of the MaterialManager class

#include "DetectorsBase/MaterialManager.h"
#include <TVirtualMC.h>  // for TVirtualMC, gMC
#include "TString.h"     // for TString
#include <iostream>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace o2::Base;

void MaterialManager::Material(const char* modname, Int_t imat, const char* name, Float_t a, Float_t z, Float_t dens,
                               Float_t radl, Float_t absl, Float_t* buf, Int_t nwbuf)
{
  TString uniquename = modname;
  uniquename.Append("_");
  uniquename.Append(name);

  // Check this!!!
  int kmat = -1;
  TVirtualMC::GetMC()->Material(kmat, uniquename.Data(), a, z, dens * mDensityFactor, radl, absl, buf, nwbuf);
  mMaterialMap[modname][imat] = kmat;
  insertMaterialName(uniquename.Data(), kmat);
  printMaterials();
}

void MaterialManager::Mixture(const char* modname, Int_t imat, const char* name, Float_t* a, Float_t* z, Float_t dens,
                              Int_t nlmat, Float_t* wmat)
{
  TString uniquename = modname;
  uniquename.Append("_");
  uniquename.Append(name);

  // Check this!!!
  int kmat = -1;
  TVirtualMC::GetMC()->Mixture(kmat, uniquename.Data(), a, z, dens * mDensityFactor, nlmat, wmat);
  mMaterialMap[modname][imat] = kmat;
  insertMaterialName(uniquename.Data(), kmat);
  printMaterials();
}

void MaterialManager::Medium(const char* modname, Int_t numed, const char* name, Int_t nmat, Int_t isvol, Int_t ifield,
                             Float_t fieldm, Float_t tmaxfd, Float_t stemax, Float_t deemax, Float_t epsil,
                             Float_t stmin, Float_t* ubuf, Int_t nbuf)
{
  TString uniquename = modname;
  uniquename.Append("_");
  uniquename.Append(name);

  // Check this!!!
  int kmed = -1;
  const int kmat = getMaterialID(modname, nmat);
  TVirtualMC::GetMC()->Medium(kmed, uniquename.Data(), kmat, isvol, ifield, fieldm, tmaxfd, stemax, deemax, epsil,
                              stmin, ubuf, nbuf);
  mMediumMap[modname][numed] = kmed;
  insertMediumName(uniquename.Data(), kmed);
}

void MaterialManager::printMaterials() const
{
  for (auto& p : mMaterialMap) {
    auto name = p.first;
    std::cout << "Materials for key " << name << "\n";
    for (auto& e : p.second) {
      std::cout << "internal id " << e.first << " to " << e.second << "\n";
    }
  }
}

void MaterialManager::printMedia() const
{
  for (auto& p : mMediumMap) {
    auto name = p.first;
    std::cout << "Materials for key " << name << "\n";
    for (auto& e : p.second) {
      std::cout << "internal id " << e.first << " to " << e.second << "\n";
    }
  }
}

// insert material name
void MaterialManager::insertMaterialName(const char* uniquename, int index)
{
  assert(mMaterialNameToGlobalIndexMap.find(uniquename) == mMaterialNameToGlobalIndexMap.end());
  mMaterialNameToGlobalIndexMap[uniquename] = index;
}

void MaterialManager::insertMediumName(const char* uniquename, int index)
{
  assert(mMediumNameToGlobalIndexMap.find(uniquename) == mMediumNameToGlobalIndexMap.end());
  mMediumNameToGlobalIndexMap[uniquename] = index;
}

ClassImp(o2::Base::MaterialManager)




