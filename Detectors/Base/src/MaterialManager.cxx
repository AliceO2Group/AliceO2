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

using namespace o2::base;

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
  { EProc::kCKOV, "CKOV" },
  { EProc::kRAYL, "RAYL" },
  { EProc::kLABS, "LABS" }
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

void MaterialManager::loadCutsAndProcessesFromFile(const char* modname, const char* filename)
{
  // Implementation of a method to set cuts and processes as done in AliRoot.
  // The file is expected to contain columns of the form
  // MODNAME LOCALMEDIUMID CUT0 ... CUT9 FLAG0 ... FLAG11
  // where cuts and flags correspond to keys denoted in cutnames and procnames below.

  const int NCUTS = 10;  // number of cut columns expected in file
  const int NFLAGS = 12; // number of process flag columns expected in file

  /// list of cut enumerated in ascending column mode as written in file
  using namespace o2::base;
  ECut cutnames[NCUTS] = { ECut::kCUTGAM,
                           ECut::kCUTELE,
                           ECut::kCUTNEU,
                           ECut::kCUTHAD,
                           ECut::kCUTMUO,
                           ECut::kBCUTE,
                           ECut::kBCUTM,
                           ECut::kDCUTE,
                           ECut::kDCUTM,
                           ECut::kPPCUTM };

  /// list of process flags enumerated in ascending column mode as written in file
  // missing STRA for the moment
  EProc procnames[NFLAGS - 1] = { EProc::kANNI,
                                  EProc::kBREM,
                                  EProc::kCOMP,
                                  EProc::kDCAY,
                                  EProc::kDRAY,
                                  EProc::kHADR,
                                  EProc::kLOSS,
                                  EProc::kMULS,
                                  EProc::kPAIR,
                                  EProc::kPHOT,
                                  EProc::kRAYL };

  std::ifstream cutfile(filename);

  if (!cutfile.is_open()) {
    LOG(ERROR) << "File " << filename << " does not exist; Cannot apply cuts";
    return;
  }

  // reading from file
  float cut[NCUTS]; // to store cut values
  int flag[NFLAGS]; // to store flags
  int itmed, iret;
  char line[256];
  char detName[7];

  while (cutfile.getline(line, 256)) {
    // Initialise cuts and flags for this line
    for (int i = 0; i < NCUTS; i++) {
      cut[i] = -99;
    }
    for (int i = 0; i < NFLAGS; i++) {
      flag[i] = -99;
    }
    if (strlen(line) == 0) {
      continue;
    }
    // ignore comments marked by *
    if (line[0] == '*') {
      continue;
    }
    // Read the numbers
    iret = sscanf(line, "%6s %d %f %f %f %f %f %f %f %f %f %f %d %d %d %d %d %d %d %d %d %d %d %d",
                  detName, &itmed, &cut[0], &cut[1], &cut[2], &cut[3], &cut[4], &cut[5], &cut[6], &cut[7], &cut[8],
                  &cut[9], &flag[0], &flag[1], &flag[2], &flag[3], &flag[4], &flag[5], &flag[6], &flag[7],
                  &flag[8], &flag[9], &flag[10], &flag[11]);
    if (!iret) {
      // nothing read
      continue;
    }

    // apply cuts via material manager interface
    for (int i = 0; i < NCUTS; ++i) {
      if (cut[i] >= 0.) {
        SpecialCut(modname, itmed, cutnames[i], cut[i]);
      }
    }

    // apply process flags
    for (int i = 0; i < NFLAGS - 1; ++i) {
      if (flag[i] >= 0) {
        SpecialProcess(modname, itmed, procnames[i], flag[i]);
      }
    }
  } // end loop over lines
}

ClassImp(o2::base::MaterialManager)
