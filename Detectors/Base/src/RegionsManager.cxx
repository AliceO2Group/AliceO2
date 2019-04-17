// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RegionsManager.cxx
/// \brief Implementation of the RegionsManager class

#include <algorithm>

#include "DetectorsBase/RegionsManager.h"
#include <FairLogger.h>

using namespace o2::base;

void RegionsManager::addRootVolumeToRegion(const std::string& regionName,const std::string& rootVolumeName)
{
  if(!checkRootVolumeExists(rootVolumeName)) {
    LOG(FATAL) << "Volume " << rootVolumeName << " is already a root volume";
  }
  insertRootVolume(regionName, rootVolumeName);
  LOG(INFO) << "Added root volume " << rootVolumeName << " to region " << regionName;
}

bool RegionsManager::checkRootVolumeExists(const std::string& rootVolumeName) const
{
  if(std::find(mRootVolumeNames.begin(), mRootVolumeNames.end(), rootVolumeName) != mRootVolumeNames.end()) {
    return false;
  }
  return true;
}

void RegionsManager::insertRootVolume(const std::string& regionName,const std::string& rootVolumeName)
{
  mRootVolumeNames.push_back(rootVolumeName);
  mRegionVolumesMap[regionName].push_back(rootVolumeName);
}

const std::map<std::string, std::vector<std::string>>&
RegionsManager::getRegionVolumesMap() const
{
  return mRegionVolumesMap;
}

ClassImp(o2::base::RegionsManager)
