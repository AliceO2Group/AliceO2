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
#include "TVirtualMC.h"
#include "TString.h" // for TString
#include <TGeoMedium.h>
#include <TGeoManager.h>
#include <TList.h>
#include <iostream>
#include <utility>
#include <FairLogger.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace o2::Base;

const std::unordered_map<EProc, const char*> MaterialManager::mProcessIDToName = {
  { EProc::kPAIR, "PAIR" },
  { EProc::kCOMP, "COMP" },
  { EProc::kPHOT, "PHOT" },
  { EProc::kPFIS, "PFIS" },
  { EProc::kDRAY, "DRAY" },
  { EProc::kANNI, "ANNI" },
  { EProc::kBREM, "BREM" },
  { EProc::kHADR, "HADR" },
  { EProc::kMUNU, "MUNU" },
  { EProc::kDCAY, "DCAY" },
  { EProc::kLOSS, "LOSS" },
  { EProc::kMULS, "MULS" },
  { EProc::kCKOV, "CKOV" }
};

const std::unordered_map<ECut, const char*> MaterialManager::mCutIDToName = {
  { ECut::kCUTGAM, "CUTGAM" },
  { ECut::kCUTELE, "CUTELE" },
  { ECut::kCUTNEU, "CUTNEU" },
  { ECut::kCUTHAD, "CUTHAD" },
  { ECut::kCUTMUO, "CUTMUO" },
  { ECut::kBCUTE, "BCUTE" },
  { ECut::kBCUTM, "BCUTM" },
  { ECut::kDCUTE, "DCUTE" },
  { ECut::kDCUTM, "DCUTM" },
  { ECut::kPPCUTM, "PPCUTM" },
  { ECut::kTOFMAX, "TOFMAX" }
};

void MaterialManager::Material(const char* modname, Int_t imat, const char* name, Float_t a, Float_t z, Float_t dens,
                               Float_t radl, Float_t absl, Float_t* buf, Int_t nwbuf)
{
  TString uniquename = modname;
  uniquename.Append("_");
  uniquename.Append(name);

  if (TVirtualMC::GetMC()) {
    // Check this!!!
    int kmat = -1;
    TVirtualMC::GetMC()->Material(kmat, uniquename.Data(), a, z, dens * mDensityFactor, radl, absl, buf, nwbuf);
    mMaterialMap[modname][imat] = kmat;
    insertMaterialName(uniquename.Data(), kmat);
  } else {
    auto uid = gGeoManager->GetListOfMaterials()->GetSize();
    auto mat = gGeoManager->Material(uniquename.Data(), a, z, dens * mDensityFactor, uid, radl, absl);
    mMaterialMap[modname][imat] = uid;
    insertMaterialName(uniquename.Data(), uid);
  }
}

/// Define a mixture or a compound
/// @param imat local (to detector/module) mixture identifier
/// @param a,z,wmat arrays of size abs(nlmat) defining the materials
/// @param nlmat indicates what wmat array represents
///
/// If nlmat > 0 then wmat contains the proportion by
/// weights of each basic material in the mixture.
///
/// If nlmat < 0 then wmat contains the number of atoms
/// of a given kind into the molecule of the compound.
/// In this case, wmat in output is changed to relative
/// weights.
void MaterialManager::Mixture(const char* modname, Int_t imat, const char* name, Float_t* a, Float_t* z, Float_t dens,
                              Int_t nlmat, Float_t* wmat)
{
  TString uniquename = modname;
  uniquename.Append("_");
  uniquename.Append(name);

  if (TVirtualMC::GetMC()) {
    // Check this!!!
    int kmat = -1;
    TVirtualMC::GetMC()->Mixture(kmat, uniquename.Data(), a, z, dens * mDensityFactor, nlmat, wmat);
    mMaterialMap[modname][imat] = kmat;
    insertMaterialName(uniquename.Data(), kmat);

  } else {
    auto uid = gGeoManager->GetListOfMaterials()->GetSize();
    if (nlmat < 0) {
      nlmat = -nlmat;
      Double_t amol = 0;
      Int_t i;
      for (i = 0; i < nlmat; i++) {
        amol += a[i] * wmat[i];
      }
      for (i = 0; i < nlmat; i++) {
        wmat[i] *= a[i] / amol;
      }
    }
    auto mix = gGeoManager->Mixture(uniquename.Data(), a, z, dens * mDensityFactor, nlmat, wmat, uid);
    mMaterialMap[modname][imat] = uid;
    insertMaterialName(uniquename.Data(), uid);
  }
}

void MaterialManager::Medium(const char* modname, Int_t numed, const char* name, Int_t nmat, Int_t isvol, Int_t ifield,
                             Float_t fieldm, Float_t tmaxfd, Float_t stemax, Float_t deemax, Float_t epsil,
                             Float_t stmin, Float_t* ubuf, Int_t nbuf)
{
  TString uniquename = modname;
  uniquename.Append("_");
  uniquename.Append(name);

  if (TVirtualMC::GetMC()) {
    // Check this!!!
    int kmed = -1;
    const int kmat = getMaterialID(modname, nmat);
    TVirtualMC::GetMC()->Medium(kmed, uniquename.Data(), kmat, isvol, ifield, fieldm, tmaxfd, stemax, deemax, epsil,
                                stmin, ubuf, nbuf);
    mMediumMap[modname][numed] = kmed;
    insertMediumName(uniquename.Data(), kmed);
    insertTGeoMedium(modname, numed);
  } else {
    auto uid = gGeoManager->GetListOfMedia()->GetSize();
    auto med = gGeoManager->Medium(uniquename.Data(), uid, getMaterialID(modname, nmat), isvol, ifield, fieldm, tmaxfd,
                                   stemax, deemax, epsil, stmin);
    mMediumMap[modname][numed] = uid;
    insertMediumName(uniquename.Data(), uid);
    insertTGeoMedium(modname, numed);
  }
}

void MaterialManager::Processes(ESpecial special, int globalindex,
                                const std::initializer_list<std::pair<EProc, int>>& parIDValMap)
{
  for (auto& m : parIDValMap) {
    Process(special, globalindex, m.first, m.second);
  }
}

void MaterialManager::Cuts(ESpecial special, int globalindex,
                           const std::initializer_list<std::pair<ECut, Float_t>>& parIDValMap)
{
  for (auto& m : parIDValMap) {
    Cut(special, globalindex, m.first, m.second);
  }
}

void MaterialManager::Cut(ESpecial special, int globalindex, ECut cut, Float_t val)
{
  // this check is needed, in principal only for G3, otherwise SegFault
  if (val < 0.) {
    return;
  }
  auto it = mCutIDToName.find(cut);
  if (it == mCutIDToName.end()) {
    return;
  }
  if (special == ESpecial::kFALSE) {
    mDefaultCutMap[cut] = val;
    /// Explicit template definition to cover this which differs from global cut setting
    TVirtualMC::GetMC()->SetCut(it->second, val);
  } else if (mApplySpecialCuts) {
    mMediumCutMap[globalindex][cut] = val;
    TVirtualMC::GetMC()->Gstpar(globalindex, it->second, val);
  }
}

void MaterialManager::Process(ESpecial special, int globalindex, EProc process, int val)
{
  // this check is needed, in principal only for G3, otherwise SegFault
  if (val < 0) {
    return;
  }
  auto it = mProcessIDToName.find(process);
  if (it == mProcessIDToName.end()) {
    return;
  }
  if (special == ESpecial::kFALSE) {
    mDefaultProcessMap[process] = val;
    /// Explicit template definition to cover this which differs from global process setting
    TVirtualMC::GetMC()->SetProcess(it->second, val);
  } else if (mApplySpecialProcesses) {
    mMediumProcessMap[globalindex][process] = val;
    TVirtualMC::GetMC()->Gstpar(globalindex, it->second, val);
  }
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

void MaterialManager::printProcesses() const
{
  LOG(INFO) << "Print process settings of media.";
  std::cout << "Default process settings:\n";
  for (auto& p : mDefaultProcessMap) {
    auto it = mProcessIDToName.find(p.first);
    if (it != mProcessIDToName.end()) {
      std::cout << "\t" << it->second << " = " << p.second << "\n";
    }
  }
  if (mApplySpecialProcesses && mMediumProcessMap.size() > 0) {
    std::cout << "Settings for single media:\n";
    for (auto& m : mMediumProcessMap) {
      std::cout << "Global medium ID " << m.first << " (module = " << getModuleFromMediumID(m.first)
                << ", medium name = " << getMediumNameFromMediumID(m.first) << "):\n";
      for (auto& p : m.second) {
        auto it = mProcessIDToName.find(p.first);
        if (it != mProcessIDToName.end()) {
          std::cout << "\t" << it->second << " = " << p.second << "\n";
        }
      }
    }
  }
}

void MaterialManager::printCuts() const
{
  LOG(INFO) << "Print cut settings of media.";
  std::cout << "Default cut settings:\n";
  for (auto& c : mDefaultCutMap) {
    auto it = mCutIDToName.find(c.first);
    if (it != mCutIDToName.end()) {
      std::cout << "\t" << it->second << " = " << c.second << "\n";
    }
  }
  if (mApplySpecialCuts && mMediumCutMap.size() > 0) {
    std::cout << "Settings for single media:\n";
    for (auto& m : mMediumCutMap) {
      std::cout << "Global medium ID " << m.first << " (module = " << getModuleFromMediumID(m.first)
                << ", medium name = " << getMediumNameFromMediumID(m.first) << "):\n";
      for (auto& c : m.second) {
        auto it = mCutIDToName.find(c.first);
        if (it != mCutIDToName.end()) {
          std::cout << "\t" << it->second << " = " << c.second << "\n";
        }
      }
    }
  }
}

// insert material name
void MaterialManager::insertMaterialName(const char* uniquename, int index)
{
  assert(mMaterialNameToGlobalIndexMap.find(uniquename) == mMaterialNameToGlobalIndexMap.end());
  mMaterialNameToGlobalIndexMap[uniquename] = index;
}

// inserts the last
void MaterialManager::insertTGeoMedium(std::string modname, int localindex)
{
  auto p = std::make_pair(modname, localindex);
  assert(mTGeoMediumMap.find(p) == mTGeoMediumMap.end());
  auto list = gGeoManager->GetListOfMedia();
  mTGeoMediumMap[p] = (TGeoMedium*)list->At(list->GetEntries() - 1);

  LOG(DEBUG) << "mapping " << modname << " " << localindex << " to " << mTGeoMediumMap[p]->GetName();
}

void MaterialManager::insertMediumName(const char* uniquename, int index)
{
  assert(mMediumNameToGlobalIndexMap.find(uniquename) == mMediumNameToGlobalIndexMap.end());
  mMediumNameToGlobalIndexMap[uniquename] = index;
}

// find TGeoMedium instance given a detector prefix and a local (vmc) medium index
TGeoMedium* MaterialManager::getTGeoMedium(std::string const& modname, int localindex)
{
  auto p = std::make_pair(modname, localindex);
  auto iter = mTGeoMediumMap.find(p);
  if (iter == mTGeoMediumMap.end()) {
    LOG(WARNING) << "No medium registered for " << modname << " index " << localindex << "\n";
    return nullptr;
  }
  return iter->second;
}

// find TGeoMedium instance given the full medium name
// mainly forwards directly to TGeoManager and raises a warning if medium is nullptr
TGeoMedium* MaterialManager::getTGeoMedium(const char* mediumname)
{
  auto med = gGeoManager->GetMedium(mediumname);
  assert(med != nullptr);
  return med;
}

ClassImp(o2::Base::MaterialManager)
