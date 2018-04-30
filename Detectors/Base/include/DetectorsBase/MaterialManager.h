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

  /// Global settings of processes.
  /// To ignore a certain process to be set explicitly, just set it to o2::Base::MaterialManager::NOPROCESS
  void Processes( int pair, int comp, int phot, int pfis, int dray, int anni, int brem, int hadr, int munu, 
                  int dcay, int loss, int muls, int ckov )
  {
    Processes( false, -1, pair, comp, phot, pfis, dray, anni, brem, hadr, munu, dcay, loss, muls, ckov );
  }
  /// Set processes per medium providing the module name and the local ID of the medium.
  /// To ignore a certain process to be set explicitly (default or Geant settings will be used in that case) use
  /// o2::Base::MaterialManager::NOPROCESS
  void Processes( const char* modname, int localindex, int pair, int comp, int phot, int pfis, int dray, int anni,
                                                       int brem, int hadr, int munu, int dcay, int loss, int muls, 
                                                       int ckov )
  {
    int globalindex = getMediumID( modname, localindex );
    if( globalindex != -1 )
    {
      Processes( true, globalindex, pair, comp, phot, pfis, dray, anni, brem, 
                                    hadr, munu, dcay, loss, muls, ckov );
    }
  }
  /// Global settings of cuts.
  /// To ignore a certain cut to be set, just set it to o2::Base::MaterialManager::NOPROCESS
  void Cuts( Float_t cutgam, Float_t cutele, Float_t cutneu, Float_t cuthad, Float_t cutmuo, Float_t bcute, 
             Float_t bcutm, Float_t dcute, Float_t dcutm,  Float_t ppcutm, Float_t tofmax )
  {
    Cuts( false, -1, cutgam, cutele, cutneu, cuthad, cutmuo, bcute, bcutm, dcute, dcutm, ppcutm, tofmax );
  }
  /// Set cuts per medium providing the module name and the local ID of the medium.
  /// To ignore a certain cut to be set explicitly (default or Geant settings will be used in that case) use
  /// o2::Base::MaterialManager::NOPROCESS
  void Cuts( const char* modname, int localindex, Float_t cutgam, Float_t cutele, Float_t cutneu, Float_t cuthad, 
                                                  Float_t cutmuo, Float_t bcute, Float_t bcutm, Float_t dcute, 
                                                  Float_t dcutm,  Float_t ppcutm, Float_t tofmax )
  {
    int globalindex = getMediumID( modname, localindex );
    if( globalindex != -1 )
    {
      Cuts( true, globalindex, cutgam, cutele, cutneu, cuthad, cutmuo, bcute, 
                               bcutm, dcute, dcutm, ppcutm, tofmax );
    }
  }
  /// Custom setting of process or cut given parameter name and value
  void Process( const char* modname, int localindex, int parID, int val )
  {
    int globalindex = getMediumID( modname, localindex );
    // do that here because user might give wrong parameter.
    if( isValidProcess( parID ) && globalindex != -1 )
    {
      Process( true, globalindex, parID, val );
    }
  }
  /// set default process
  void Process( int parID, int val )
  {
    // do that here because user might give wrong parameter.
    if( isValidProcess( parID ) )
    {
      Process( false, -1, parID, val );
    }
  }
  /// Custom setting of process or cut given parameter name and value
  void Cut( const char* modname, int localindex, int parID, Float_t val )
  {
    int globalindex = getMediumID( modname, localindex );
    // do that here because user might give wrong parameter.
    if( isValidCut( parID ) && globalindex != -1 )
    {
      Cut( true, globalindex, parID, val );
    }
  }
  /// set default cut
  void Cut( int parID, Float_t val )
  {
    // do that here because user might give wrong parameter.
    if( isValidCut( parID ) )
    {
      Cut( false, -1, parID, val );
    }
  }
 private:
  // Hide details by providing these private methods so it cannot happen that special settings 
  // are applied as default settings by accident using a boolean flag
  void Processes( bool special, int globalindex, int pair, int comp, int phot, int pfis, int dray, int anni, 
                                                 int brem, int hadr, int munu, int dcay, int loss, int muls, 
                                                 int ckov );
  void Cuts( bool special, int globalindex, Float_t cutgam, Float_t cutele, Float_t cutneu, Float_t cuthad, 
                                            Float_t cutmuo, Float_t bcute, Float_t bcutm, Float_t dcute, 
                                            Float_t dcutm, Float_t ppcutm, Float_t tofmax );
  void Process( bool special, int globalindex, int parID, int val );
  void Cut( bool special, int globalindex, int parID, Float_t val );
  
  // insert material name
  void insertMaterialName(const char* uniquename, int index);
  void insertMediumName(const char* uniquename, int index);
  void insertTGeoMedium(std::string modname, int localindex);

 public:
  /// Set flags whether to use special cuts and process settings
  void applySpecialProcesses( bool val = true )
  {
    mApplySpecialProcesses = val;
  }
  void applySpecialCuts( bool val = true )
  {
    mApplySpecialCuts = val;
  }

  /// Check whether it is a valid process parameter
  bool isValidProcess( int parID ) const
  {
    if( parID >= mProcessIDToName.size() )
    {
      return false;
    }
    return true;
  }

  /// Check whether it is a valid cut parameter
  bool isValidCut( int parID ) const
  {
    if( parID >= mCutIDToName.size() )
    {
      return false;
    }
    return true;
  }

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

  std::map< int, std::map< int, int >> mMediumProcessMap; // map of global medium id to parameter-value map of processes
  std::map< int, std::map< int, Float_t >> mMediumCutMap; // map of global medium id to parameter-value map of cuts
  std::map< int, Float_t > mDefaultCutMap; // map of global cuts
  std::map< int, int > mDefaultProcessMap; // map of global processes

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
  const static std::vector< std::string > mCutIDToName;
  /// fixed names of processes
  const static std::vector< std::string > mProcessIDToName;

  /// Decide whether special process and cut settings should be applied
  bool mApplySpecialProcesses = true;
  bool mApplySpecialCuts = true;
  
 public:
  /// processes available
  enum { PAIR = 0, COMP, PHOT, PFIS, DRAY, ANNI, BREM, HADR, MUNU, DCAY, LOSS, MULS, CKOV };
  /// cuts available
  enum { CUTGAM = 0, CUTELE, CUTNEU, CUTHAD, CUTMUO, BCUTE, BCUTM, DCUTE, DCUTM, PPCUTM, TOFMAX };
  /// can be used to ignore cut setting for a medium
  constexpr static Float_t NOCUT = -1.;
  /// can be used to ignore the process setting for a medium
  constexpr static int NOPROCESS = -1;

  ClassDefNV(MaterialManager, 0);
};
}
}

#endif
