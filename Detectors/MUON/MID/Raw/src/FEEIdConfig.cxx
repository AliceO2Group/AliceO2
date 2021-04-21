// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/FEEIdConfig.cxx
/// \brief  Hardware Id to FeeId mapper
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 March 2020

#include "MIDRaw/FEEIdConfig.h"

#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include "Framework/Logger.h"
#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{
FEEIdConfig::FEEIdConfig() : mLinkUniqueIdToGBTUniqueId(), mGBTUniqueIdToFeeId(), mGBTUniqueIdsInLink()
{
  /// Default constructor
  for (uint16_t iside = 0; iside < 2; ++iside) {
    uint16_t cruId = iside;
    for (uint8_t igbt = 0; igbt < crateparams::sNGBTsPerSide; ++igbt) {
      uint8_t epId = igbt / 8;
      uint16_t feeId = 2 * cruId + epId;
      uint16_t gbtUniqueId = igbt + crateparams::sNGBTsPerSide * iside;
      mLinkUniqueIdToGBTUniqueId[getLinkUniqueId(igbt % 8, epId, iside)] = gbtUniqueId;
      mGBTUniqueIdToFeeId[gbtUniqueId] = feeId;
      mGBTUniqueIdsInLink[feeId].emplace_back(gbtUniqueId);
    }
  }
}

FEEIdConfig::FEEIdConfig(const char* filename) : mLinkUniqueIdToGBTUniqueId()
{
  /// Construct from file
  load(filename);
}

uint16_t FEEIdConfig::getGBTUniqueId(uint32_t linkUniqueId) const
{
  /// Gets the feeId from the physical ID of the link
  auto feeId = mLinkUniqueIdToGBTUniqueId.find(linkUniqueId);
  if (feeId == mLinkUniqueIdToGBTUniqueId.end()) {
    LOGF(ERROR, "No FeeId found for: CRUId: %i  LinkId: %i  EndPointId: %i", getCRUId(linkUniqueId), getLinkId(linkUniqueId), getEndPointId(linkUniqueId));
    return 0xFFFF;
  }
  return feeId->second;
}

std::vector<uint16_t> FEEIdConfig::getConfiguredGBTUniqueIDs() const
{
  /// Returns the sorted list of configured FEE IDs
  std::vector<uint16_t> configIds;
  for (auto& item : mLinkUniqueIdToGBTUniqueId) {
    configIds.emplace_back(item.second);
  }
  std::sort(configIds.begin(), configIds.end());
  return configIds;
}

std::vector<uint32_t> FEEIdConfig::getConfiguredLinkUniqueIDs() const
{
  /// Returns the sorted list of configured GBT IDs
  std::vector<uint32_t> configIds;
  for (auto& item : mLinkUniqueIdToGBTUniqueId) {
    configIds.emplace_back(item.first);
  }
  std::sort(configIds.begin(), configIds.end());
  return configIds;
}

std::vector<uint16_t> FEEIdConfig::getConfiguredFEEIDs() const
{
  /// Returns the sorted list of configured FEE Ids
  std::vector<uint16_t> configIds;
  for (auto& item : mGBTUniqueIdsInLink) {
    configIds.emplace_back(item.first);
  }
  std::sort(configIds.begin(), configIds.end());
  return configIds;
}

bool FEEIdConfig::load(const char* filename)
{
  /// Loads the FEE Ids from a configuration file
  /// The file is in the form:
  /// feeId linkId endPointId cruId
  /// with one line per link

  mLinkUniqueIdToGBTUniqueId.clear();
  mGBTUniqueIdToFeeId.clear();
  mGBTUniqueIdsInLink.clear();
  std::ifstream inFile(filename);
  if (!inFile.is_open()) {
    return false;
  }
  std::string line, token;
  while (std::getline(inFile, line)) {
    int nSpaces = std::count(line.begin(), line.end(), ' ');
    if (nSpaces < 3) {
      continue;
    }
    if (line.find('#') < line.find(' ')) {
      continue;
    }
    std::stringstream ss;
    ss << line;
    std::getline(ss, token, ' ');
    uint16_t gbtUniqueId = std::atoi(token.c_str());
    std::getline(ss, token, ' ');
    uint8_t linkId = std::atoi(token.c_str());
    std::getline(ss, token, ' ');
    uint8_t endPointId = std::atoi(token.c_str());
    std::getline(ss, token, ' ');
    uint16_t cruId = std::atoi(token.c_str());
    std::getline(ss, token, ' ');
    uint16_t feeId = 2 * cruId + endPointId;
    if (nSpaces > 3) {
      feeId = std::atoi(token.c_str());
      std::getline(ss, token, ' ');
    }
    mLinkUniqueIdToGBTUniqueId[getLinkUniqueId(linkId, endPointId, cruId)] = gbtUniqueId;
    mGBTUniqueIdToFeeId[gbtUniqueId] = feeId;
    mGBTUniqueIdsInLink[feeId].emplace_back(gbtUniqueId);
  }
  inFile.close();
  return true;
}

void FEEIdConfig::write(const char* filename) const
{
  /// Writes the FEE Ids to a configuration file
  std::ofstream outFile(filename);
  for (auto& id : mLinkUniqueIdToGBTUniqueId) {
    outFile << id.second << " " << getLinkId(id.first) << " " << getEndPointId(id.first) << " " << getCRUId(id.first) << " " << mGBTUniqueIdToFeeId.find(id.second)->second << std::endl;
  }
  outFile.close();
}

} // namespace mid
} // namespace o2
