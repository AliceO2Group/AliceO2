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
#include <set>

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
    std::cout << "Tracking media for key " << name << "\n";
    for (auto& e : p.second) {
      auto medname = getMediumNameFromMediumID(e.second);
      std::cout << medname << " ";
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
    LOG(WARN) << "File " << filename << " does not exist; Cannot apply cuts";
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

/// Set cuts per medium providing the module name and the local ID of the medium.
/// To ignore a certain cut to be set explicitly (default or Geant settings will be used in that case) use
/// o2::base::MaterialManager::NOPROCESS
void MaterialManager::SpecialCuts(const char* modname, int localindex,
                                  const std::initializer_list<std::pair<ECut, Float_t>>& parIDValMap)
{
  int globalindex = getMediumID(modname, localindex);
  if (globalindex != -1) {
    Cuts(ESpecial::kTRUE, globalindex, parIDValMap);
  }
}

void MaterialManager::SpecialCut(const char* modname, int localindex, ECut parID, Float_t val)
{
  int globalindex = getMediumID(modname, localindex);
  if (globalindex != -1) {
    Cut(ESpecial::kTRUE, globalindex, parID, val);
  }
}

/// Custom setting of process or cut given parameter name and value
void MaterialManager::SpecialProcess(const char* modname, int localindex, EProc parID, int val)
{
  int globalindex = getMediumID(modname, localindex);
  if (globalindex != -1) {
    Process(ESpecial::kTRUE, globalindex, parID, val);
  }
}

int MaterialManager::getMaterialID(const char* modname, int imat) const
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

int MaterialManager::getMediumID(const char* modname, int imed) const
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
void MaterialManager::getMediumIDMappingAsVector(const char* modname, std::vector<int>& mapping) const
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

void MaterialManager::getMediaWithSpecialProcess(EProc process, std::vector<int>& mediumProcessVector) const
{
  // clear
  mediumProcessVector.clear();
  // resize to maximum number of global IDs for which special processes are set. In case process is not
  // implemented for a certain medium, value is -1
  mediumProcessVector.resize(mMediumProcessMap.size(), -1);
  // find media
  for (auto& m : mMediumProcessMap) {
    // loop over processes in medium
    for (auto& p : m.second) {
      // push medium ID if process is there
      if (p.first == process) {
        mediumProcessVector[m.first] = p.second;
        break;
      }
    }
  }
}

void MaterialManager::getMediaWithSpecialCut(ECut cut, std::vector<Float_t>& mediumCutVector) const
{
  // clear
  mediumCutVector.clear();
  // resize to maximum number of global IDs for which special cuts are set. In case cut is not implemented
  // for a certain medium, value is -1.
  mediumCutVector.resize(mMediumCutMap.size(), -1.);
  // find media
  for (auto& m : mMediumCutMap) {
    // loop over cuts in medium
    for (auto& c : m.second) {
      // push medium ID if cut is there
      if (c.first == cut) {
        mediumCutVector[m.first] = c.second;
        break;
      }
    }
  }
}

/// Fill vector with default processes
void MaterialManager::getDefaultProcesses(std::vector<std::pair<EProc, int>>& processVector)
{
  processVector.clear();
  for (auto& m : mDefaultProcessMap) {
    processVector.emplace_back(m.first, m.second);
  }
}
/// Fill vector with default cuts
void MaterialManager::getDefaultCuts(std::vector<std::pair<ECut, Float_t>>& cutVector)
{
  cutVector.clear();
  for (auto& m : mDefaultCutMap) {
    cutVector.emplace_back(m.first, m.second);
  }
}
/// Get special processes for global medium ID
void MaterialManager::getSpecialProcesses(int globalindex, std::vector<std::pair<EProc, int>>& processVector)
{
  processVector.clear();
  if (mMediumProcessMap.find(globalindex) != mMediumProcessMap.end()) {
    for (auto& m : mMediumProcessMap[globalindex]) {
      processVector.emplace_back(m.first, m.second);
    }
  }
}
/// Interface for module name and local medium ID
void MaterialManager::getSpecialProcesses(const char* modname, int localindex, std::vector<std::pair<EProc, int>>& processVector)
{
  int globalindex = getMediumID(modname, localindex);
  if (globalindex != -1) {
    getSpecialProcesses(globalindex, processVector);
  }
}
/// Get special cuts for global medium ID
void MaterialManager::getSpecialCuts(int globalindex, std::vector<std::pair<ECut, Float_t>>& cutVector)
{
  cutVector.clear();
  if (mMediumCutMap.find(globalindex) != mMediumCutMap.end()) {
    for (auto& m : mMediumCutMap[globalindex]) {
      cutVector.emplace_back(m.first, m.second);
    }
  }
}
/// Interface for module name and local medium ID
void MaterialManager::getSpecialCuts(const char* modname, int localindex, std::vector<std::pair<ECut, Float_t>>& cutVector)
{
  int globalindex = getMediumID(modname, localindex);
  if (globalindex != -1) {
    getSpecialCuts(globalindex, cutVector);
  }
}

const char* MaterialManager::getModuleFromMediumID(int globalindex) const
{
  // loop over module names and corresponding local<->global mapping
  for (auto& m : mMediumMap) {
    for (auto& i : m.second) {
      // is the global index there?
      if (i.second == globalindex) {
        // \note maybe unsafe in case mMediumMap is altered in the same scope where the returned C string is used
        // since that points to memory of string it was derived from.
        return m.first.c_str();
      }
    }
  }
  // module is UNKNOWN if global medium ID could not be found.
  return "UNKNOWN";
}

/// Get medium name from global medium ID
const char* MaterialManager::getMediumNameFromMediumID(int globalindex) const
{
  // Get the name of the medium.
  // TODO: avoid linear search
  for (auto& n : mMediumNameToGlobalIndexMap) {
    if (n.second == globalindex) {
      // \note maybe unsafe in case mMediumMap is altered in the same scope where the returned C string is used since
      // that points to memory of string it was derived from.
      return n.first.c_str();
    }
  }
  return "UNKNOWN";
}

/// print all tracking media inside a logical volume (specified by name)
/// and all of its daughters
void MaterialManager::printContainingMedia(std::string const& volumename)
{
  auto vol = gGeoManager->FindVolumeFast(volumename.c_str());
  if (vol == nullptr) {
    LOG(WARN) << "No volume found; Cannot query medias";
  }
  std::set<TGeoMedium const*> media;

  // a little well encapsulated helper to code the recursive visitor
  // pure lambda cannot be recursive
  std::function<void(TGeoVolume const* vol, std::set<TGeoMedium const*>& mediumset)> recursivevisitor;
  recursivevisitor = [&recursivevisitor](TGeoVolume const* vol, std::set<TGeoMedium const*>& mediumset) {
    // exclude assemblies
    if (!vol->IsAssembly()) {
      mediumset.insert(vol->GetMedium());
    }
    const int n = vol->GetNdaughters();
    for (int i = 0; i < n; ++i) {
      auto daughter = vol->GetNode(i)->GetVolume();
      recursivevisitor(daughter, mediumset);
    }
  };
  recursivevisitor(vol, media);

  // simply print the media
  for (auto m : media) {
    std::cout << m->GetName() << "\n";
  }
}

ClassImp(o2::base::MaterialManager)
