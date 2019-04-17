// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RegionsManager.h
/// \brief Definition of the RegionsManager class

#ifndef ALICEO2_BASE_REGIONSMANAGER_H_
#define ALICEO2_BASE_REGIONSMANAGER_H_

#include "Rtypes.h"
#include <string>
#include <map>
#include <vector>

namespace o2
{
namespace base
{

// Central class managing creation of volume regions. It adapts the concept of
// regions as it used in GEANT4. A region is declared via its unique name and
// the name of the targeted hierachical root volume in order to be independent
// of geometry/engine implementations at this point.
class RegionsManager
{
 public:
  static RegionsManager& Instance()
  {
    static RegionsManager inst;
    return inst;
  }

  // Declare a TGeoVolume to be a root of a region
  // NOTE Does currently only work with TGeo geometry (as it used by O2)
  // \param rootVolumeName: name of the volume which should be the root volume of the region.
  //                        Must be unique! A volume cannot be a root volume of two regions
  // \param regionName: name of the region, the volume should be added to
  void addRootVolumeToRegion(const std::string& regionName,const std::string& rootVolumeName);

  // Return the regions with their root volume names
  const std::map<std::string, std::vector<std::string>>&
  getRegionVolumesMap() const;

 private:
   // Check for existing region names and duplications for rootTGeoVolName as being root volume
   // for multiple regions
   bool checkRootVolumeExists(const std::string& rootVolumeName) const;
   // Insert the region - volume pair
   // TODO This needs to be done recursively
   void insertRootVolume(const std::string& regionName,const std::string& rootVolumeName);

 private:
  RegionsManager() = default;

  // lookup structures
  // map of region name -> name of root volume
  std::map<std::string, std::vector<std::string>> mRegionVolumesMap;
  // all root volume names, must be unique, one volume cannot belong to multiple regions
  std::vector<std::string> mRootVolumeNames;

 public:
  ClassDefNV(RegionsManager, 0);
};
} // namespace base
} // namespace o2

#endif
