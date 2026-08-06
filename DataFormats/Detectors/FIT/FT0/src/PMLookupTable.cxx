// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsFT0/PMLookupTable.h"

#include "DataFormatsFT0/LookUpTable.h"
#include "Framework/Logger.h"

#include <algorithm>
#include <charconv>
#include <map>
#include <limits>
#include <string>
#include <system_error>

namespace o2::ft0
{

const PMLookupTable& PMLookupTable::Instance()
{
  static const PMLookupTable table;
  return table;
}

PMLookupTable::PMLookupTable()
{
  std::map<std::string, PMHash> moduleName2Hash;

  auto lut = SingleLUT::Instance().getVecMetadataFEE();
  std::sort(lut.begin(), lut.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.mModuleName < rhs.mModuleName;
  });

  unsigned int nextHash = 0;

  for (const auto& entry : lut) {
    const auto& moduleName = entry.mModuleName;
    const auto& moduleType = entry.mModuleType;
    const auto& channelString = entry.mChannelID;

    if (nextHash > static_cast<unsigned int>(std::numeric_limits<PMHash>::max())) {
      LOG(fatal) << "Too many FT0 FEE modules to represent with PMHash";
    }

    auto [moduleIt, inserted] = moduleName2Hash.emplace(moduleName, static_cast<PMHash>(nextHash));

    if (inserted) {
      const auto moduleHash = moduleIt->second;
      if (moduleName.find("PMA") != std::string::npos) {
        mPMHash2IsASide.emplace(moduleHash, true);
      } else if (moduleName.find("PMC") != std::string::npos) {
        mPMHash2IsASide.emplace(moduleHash, false);
      } else if (moduleType != "TCM") {
        LOG(fatal) << "Unknown FT0 module in LUT: " << moduleName
                   << " (type " << moduleType << ")";
      }
      ++nextHash;
    }

    int channelID = -1;
    const char* begin = channelString.data();
    const char* end = begin + channelString.size();
    const auto [ptr, error] = std::from_chars(begin, end, channelID);
    const bool isNumericChannel = error == std::errc{} && ptr == end;

    if (isNumericChannel) {
      if (channelID < 0 || channelID >= Constants::sNCHANNELS_PM) {
        LOG(fatal) << "Incorrect FT0 LUT entry: channel " << channelString
                   << " | module " << moduleName;
      }

      mChannelID2PMHash[channelID] = moduleIt->second;
      mChannelIsMapped[channelID] = true;
    } else if (moduleType != "TCM") {
      LOG(fatal) << "Non-TCM FT0 module without numerical channel ID: channel "
                 << channelString << " | module " << moduleName;
    }
  }

  for (ChannelID channelID = 0; channelID < Constants::sNCHANNELS_PM; ++channelID) {
    if (!mChannelIsMapped[channelID]) {
      LOG(fatal) << "FT0 channel " << channelID << " is not mapped to a PM in the LUT";
    }
  }
}

PMLookupTable::PMHash PMLookupTable::getPMHash(ChannelID channelID) const
{
  if (channelID >= Constants::sNCHANNELS_PM) {
    LOG(fatal) << "FT0 channel ID outside valid range: " << channelID;
  }
  if (!mChannelIsMapped[channelID]) {
    LOG(fatal) << "FT0 channel is not mapped to a PM: " << channelID;
  }
  return mChannelID2PMHash[channelID];
}

bool PMLookupTable::isASide(PMHash pmHash) const
{
  const auto it = mPMHash2IsASide.find(pmHash);
  if (it == mPMHash2IsASide.end()) {
    LOG(fatal) << "Unknown FT0 PM hash: " << static_cast<unsigned int>(pmHash);
  }
  return it->second;
}

} // namespace o2::ft0
