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
#include <initializer_list>

class TGeoMedium;

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

  /// In Geant3/4 there is the possibility to set custom production cuts and to enable/disable certain processes.
  /// This can be done globally as well as for each medium separately. Hence, for both cases there is one method
  /// to set default processes and cuts and another 2 methods to set cuts and processes per medium. In any case,
  /// the respective cut/process setting method is a wrapper around a private, more general, method.
  /// processes available
  enum EProc { kPAIR = 0, kCOMP, kPHOT, kPFIS, kDRAY, kANNI, kBREM, kHADR, kMUNU, kDCAY, kLOSS, kMULS, kCKOV };
  /// cuts available
  enum ECut { kCUTGAM = 0, kCUTELE, kCUTNEU, kCUTHAD, kCUTMUO, kBCUTE, kBCUTM, kDCUTE, kDCUTM, kPPCUTM, kTOFMAX };

  /// Global settings of processes.
  /// To ignore a certain process to be set explicitly, just set it to o2::Base::MaterialManager::NOPROCESS
  void DefaultProcesses(const std::initializer_list<std::pair<EProc, int>>& parIDValMap)
  {
    Processes(false, -1, parIDValMap);
  }
  /// Set processes per medium providing the module name and the local ID of the medium.
  /// To ignore a certain process to be set explicitly (default or Geant settings will be used in that case) use
  /// o2::Base::MaterialManager::NOPROCESS
  void SpecialProcesses(const char* modname, int localindex,
                        const std::initializer_list<std::pair<EProc, int>>& parIDValMap)
  {
    int globalindex = getMediumID(modname, localindex);
    if (globalindex != -1) {
      Processes(true, globalindex, parIDValMap);
    }
  }
  /// set default process
  void DefaultProcess(EProc parID, int val) { Process(false, -1, parID, val); }
  /// Custom setting of process or cut given parameter name and value
  void SpecialProcess(const char* modname, int localindex, EProc parID, int val)
  {
    int globalindex = getMediumID(modname, localindex);
    if (globalindex != -1) {
      Process(true, globalindex, parID, val);
    }
  }
  /// Global settings of cuts.
  /// To ignore a certain cut to be set, just set it to o2::Base::MaterialManager::NOPROCESS
  void DefaultCuts(const std::initializer_list<std::pair<ECut, Float_t>>& parIDValMap) { Cuts(false, -1, parIDValMap); }
  /// Set cuts per medium providing the module name and the local ID of the medium.
  /// To ignore a certain cut to be set explicitly (default or Geant settings will be used in that case) use
  /// o2::Base::MaterialManager::NOPROCESS
  void SpecialCuts(const char* modname, int localindex,
                   const std::initializer_list<std::pair<ECut, Float_t>>& parIDValMap)
  {
    int globalindex = getMediumID(modname, localindex);
    if (globalindex != -1) {
      Cuts(true, globalindex, parIDValMap);
    }
  }
  /// set default cut
  void DefaultCut(ECut parID, Float_t val) { Cut(false, -1, parID, val); }
  /// Custom setting of process or cut given parameter name and value
  void SpecialCut(const char* modname, int localindex, ECut parID, Float_t val)
  {
    int globalindex = getMediumID(modname, localindex);
    if (globalindex != -1) {
      Cut(true, globalindex, parID, val);
    }
  }

 private:
  // Hide details by providing these private methods so it cannot happen that special settings
  // are applied as default settings by accident using a boolean flag
  void Processes(bool special, int globalindex, const std::initializer_list<std::pair<EProc, int>>& parIDValMap);
  void Cuts(bool special, int globalindex, const std::initializer_list<std::pair<ECut, Float_t>>& parIDValMap);
  void Process(bool special, int globalindex, EProc parID, int val);
  void Cut(bool special, int globalindex, ECut parID, Float_t val);

  // insert material name
  void insertMaterialName(const char* uniquename, int index);
  void insertMediumName(const char* uniquename, int index);
  void insertTGeoMedium(std::string modname, int localindex);

 public:
  /// Set flags whether to use special cuts and process settings
  void enableSpecialProcesses(bool val = true) { mApplySpecialProcesses = val; }
  bool specialProcessesEnabled() const { return mApplySpecialProcesses; }
  void enableSpecialCuts(bool val = true) { mApplySpecialCuts = val; }
  bool specialCutsEnabled() const { return mApplySpecialCuts; }

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

  // various methods to get the TGeoMedium instance
  TGeoMedium* getTGeoMedium(const std::string& modname, int localid);
  TGeoMedium* getTGeoMedium(const char* mediumname);

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

  std::map<int, std::map<EProc, int>> mMediumProcessMap; // map of global medium id to parameter-value map of processes
  std::map<int, std::map<ECut, Float_t>> mMediumCutMap;  // map of global medium id to parameter-value map of cuts
  std::map<ECut, Float_t> mDefaultCutMap;                // map of global cuts
  std::map<EProc, int> mDefaultProcessMap;               // map of global processes

  // a map allowing to lookup TGeoMedia from detector name and local medium index
  std::map<std::pair<std::string, int>, TGeoMedium*> mTGeoMediumMap;

  // finally, I would like to keep track of tracking parameters and processes activated per medium

  std::map<std::string, int> mMaterialNameToGlobalIndexMap; // map of unique material name to global index
  std::map<std::string, int> mMediumNameToGlobalIndexMap;

  Float_t mDensityFactor = 1.; //! factor that is multiplied to all material densities (ONLY for
  // systematic studies)

  /// In general, transport cuts and processes are properties of detector media. On the other hand different
  /// engines might provide different cuts and processes. Further, the naming convention might differ among
  /// engines.
  /// This must be handled by the MaterialManager to fit to the engine in use. In that way, the user does not need
  /// to care about the engine in use but only needs to set cuts according to ONE naming scheme.
  // \note Currently, the naming convention of GEANT4 v10.3.3 is used.
  // \note This might be overhead so far but makes the MaterialManager and therefore O2 finally capable of
  // forwarding cuts/processe to arbitrary engines.
  // \todo Is there a more elegant implementation?
  /// fixed names of cuts
  const static std::vector<std::string> mCutIDToName;
  /// fixed names of processes
  const static std::vector<std::string> mProcessIDToName;

  /// Decide whether special process and cut settings should be applied
  bool mApplySpecialProcesses = true;
  bool mApplySpecialCuts = true;

 public:
  ClassDefNV(MaterialManager, 0);
};
} // namespace Base
} // namespace o2

#endif
