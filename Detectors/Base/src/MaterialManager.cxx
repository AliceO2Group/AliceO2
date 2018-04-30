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

const std::vector< std::string > MaterialManager::mProcessIDToName = { "PAIR", "COMP", "PHOT", "PFIS", "DRAY", 
                                                                       "ANNI", "BREM", "HADR", "MUNU", "DCAY", 
                                                                       "LOSS", "MULS", "CKOV" };

const std::vector< std::string > MaterialManager::mCutIDToName = { "CUTGAM", "CUTELE", "CUTNEU", "CUTHAD", 
                                                                   "CUTMUO", "BCUTE", "BCUTM", "DCUTE", "DCUTM", 
                                                                   "PPCUTM", "TOFMAX" };

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
  insertTGeoMedium(modname, numed);
}

void MaterialManager::Processes( bool special, int globalindex, int pair, int comp, int phot, int pfis, int dray, 
                                                                int anni, int brem, int hadr, int munu, int dcay, 
                                                                int loss, int muls, int ckov )
{

  Process( special, globalindex, PAIR, pair );
  Process( special, globalindex, COMP, comp );
  Process( special, globalindex, PHOT, phot );
  Process( special, globalindex, PFIS, pfis );
  Process( special, globalindex, DRAY, dray );
  Process( special, globalindex, ANNI, anni );
  Process( special, globalindex, BREM, brem );
  Process( special, globalindex, HADR, hadr );
  Process( special, globalindex, MUNU, munu );
  Process( special, globalindex, DCAY, dcay );
  Process( special, globalindex, LOSS, loss );
  Process( special, globalindex, MULS, muls );
  Process( special, globalindex, CKOV, ckov );
}

void MaterialManager::Cuts( bool special, int globalindex, Float_t cutgam, Float_t cutele, Float_t cutneu, 
                                                           Float_t cuthad, Float_t cutmuo, Float_t bcute, 
                                                           Float_t bcutm, Float_t dcute, Float_t dcutm, 
                                                           Float_t ppcutm, Float_t tofmax )
{
  Cut( special, globalindex, CUTGAM, cutgam );
  Cut( special, globalindex, CUTELE, cutgam );
  Cut( special, globalindex, CUTNEU, cutneu );
  Cut( special, globalindex, CUTHAD, cuthad );
  Cut( special, globalindex, CUTMUO, cutmuo );
  Cut( special, globalindex, BCUTE, bcute );
  Cut( special, globalindex, BCUTM, bcutm );
  Cut( special, globalindex, DCUTE, dcute );
  Cut( special, globalindex, DCUTM, dcutm );
  Cut( special, globalindex, PPCUTM, ppcutm );
  Cut( special, globalindex, TOFMAX, tofmax );
}

void MaterialManager::Cut( bool special, int globalindex, int parID, Float_t val )
{
  // this check is needed, in principal only for G3, otherwise SegFault
  if( val < 0. )
  {
    return;
  }
  
  if( !special )
  {
    mDefaultCutMap[ parID ] = val;
    /// Explicit template definition to cover this which differs from global cut setting
    TVirtualMC::GetMC()->SetCut( mCutIDToName[parID].c_str(), val );
  }
  else if( mApplySpecialCuts )
  {
    mMediumCutMap[ globalindex ][ parID ] = val;
    TVirtualMC::GetMC()->Gstpar( globalindex, mCutIDToName[parID].c_str(), val );
  }
  else
  {
    LOG(INFO) << "Special cuts for media are disabled.";
  }
}

void MaterialManager::Process( bool special, int globalindex, int parID, int val )
{
  // this check is needed, in principal only for G3, otherwise SegFault
  if( val < 0 )
  {
    return;
  }

  if( !special )
  {
    mDefaultProcessMap[ parID ] = val;
    /// Explicit template definition to cover this which differs from global process setting
    TVirtualMC::GetMC()->SetProcess( mProcessIDToName[parID].c_str(), val );
  }
  else if( mApplySpecialProcesses )
  {
    mMediumProcessMap[globalindex][ parID ] = val;
    TVirtualMC::GetMC()->Gstpar( globalindex, mProcessIDToName[parID].c_str(), val );
  }
  else
  {
    LOG(INFO) << "Special processes for media are disabled.";
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




