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
FEEIdConfig::FEEIdConfig() : mGBTIdToFeeId()
{
  /// Default constructor
  for (uint16_t iside = 0; iside < 2; ++iside) {
    for (uint8_t igbt = 0; igbt < crateparams::sNGBTsPerSide; ++igbt) {
      mGBTIdToFeeId[getGBTId(igbt % 12, igbt / 12, iside)] = igbt + crateparams::sNGBTsPerSide * iside;
    }
  }
}

FEEIdConfig::FEEIdConfig(const char* filename) : mGBTIdToFeeId()
{
  /// Construct from file
  load(filename);
}

uint16_t FEEIdConfig::getFeeId(uint32_t uniqueId) const
{
  /// Gets the feeId from the physical ID of the link
  auto feeId = mGBTIdToFeeId.find(uniqueId);
  if (feeId == mGBTIdToFeeId.end()) {
    LOGF(ERROR, "No FeeId found for: CRUId: %i  LinkId: %i  EndPointId: %i", getCRUId(uniqueId), getLinkId(uniqueId), getEndPointId(uniqueId));
    return 0xFFFF;
  }
  return feeId->second;
}

std::vector<uint16_t> FEEIdConfig::getConfiguredFeeIds() const
{
  /// Returns the sorted list of configured FEE IDs
  std::vector<uint16_t> configIds;
  for (auto& item : mGBTIdToFeeId) {
    configIds.emplace_back(item.second);
  }
  std::sort(configIds.begin(), configIds.end());
  return configIds;
}

std::vector<uint32_t> FEEIdConfig::getConfiguredGBTIds() const
{
  /// Returns the sorted list of configured GBT IDs
  std::vector<uint32_t> configIds;
  for (auto& item : mGBTIdToFeeId) {
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

  mGBTIdToFeeId.clear();
  std::ifstream inFile(filename);
  if (!inFile.is_open()) {
    return false;
  }
  std::string line, token;
  while (std::getline(inFile, line)) {
    if (std::count(line.begin(), line.end(), ' ') < 3) {
      continue;
    }
    if (line.find('#') < line.find(' ')) {
      continue;
    }
    std::stringstream ss;
    ss << line;
    std::getline(ss, token, ' ');
    uint16_t feeId = std::atoi(token.c_str());
    std::getline(ss, token, ' ');
    uint8_t linkId = std::atoi(token.c_str());
    std::getline(ss, token, ' ');
    uint8_t endPointId = std::atoi(token.c_str());
    std::getline(ss, token, ' ');
    uint16_t cruId = std::atoi(token.c_str());
    mGBTIdToFeeId[getGBTId(linkId, endPointId, cruId)] = feeId;
  }
  inFile.close();
  return true;
}

void FEEIdConfig::write(const char* filename) const
{
  /// Writes the FEE Ids to a configuration file
  std::ofstream outFile(filename);
  for (auto& id : mGBTIdToFeeId) {
    outFile << id.first << " " << (id.second & 0xFF) << " " << ((id.second >> 8) & 0xFF) << " " << ((id.second >> 16) & 0xFF) << std::endl;
  }
  outFile.close();
}

} // namespace mid
} // namespace o2
