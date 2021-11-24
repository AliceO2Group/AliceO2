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

/// \file   MIDRaw/ROBoardConfig.h
/// \brief  Handler for readout local board configuration
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   12 November 2021
#ifndef O2_MID_ROBOARDCONFIGHANDLER_H
#define O2_MID_ROBOARDCONFIGHANDLER_H

#include <vector>
#include <unordered_map>
#include "MIDRaw/ROBoardConfig.h"

namespace o2
{
namespace mid
{
class ROBoardConfigHandler
{
 public:
  /// Default constructor
  ROBoardConfigHandler();
  /// Constructor from file
  ROBoardConfigHandler(const char* filename);
  /// Constructor from list of local board configuration
  ROBoardConfigHandler(const std::vector<ROBoardConfig>& configurations);
  /// Default destructor
  ~ROBoardConfigHandler() = default;

  /// Returns the configuration for the local board
  const ROBoardConfig getConfig(uint8_t uniqueLocId) const;

  /// Returns the configuration map
  const std::unordered_map<uint8_t, ROBoardConfig> getConfigMap() const { return mROBoardConfigs; }

  /// Sets the local board configurations from a vector
  void set(const std::vector<ROBoardConfig>& configurations);

  /// Updates the mask values
  void updateMasks(const std::vector<ROBoard>& masks);

  /// Writes the configuration to file
  void write(const char* filename) const;

 private:
  /// Loads the board  from a configuration file
  /// The file is in the form:
  /// locId status maskX1Y1 maskX2Y2 maskX3Y3 maskX4Y4
  /// with one line per local board
  bool load(const char* filename);
  std::unordered_map<uint8_t, ROBoardConfig> mROBoardConfigs; /// Vector of local board configuration
};

/// Creates the default local board configurations
std::vector<ROBoardConfig> makeDefaultROBoardConfig(uint16_t gbtUniqueId = 0xFFFF);
/// Creates a local board configuration where no zero suppression is required
std::vector<ROBoardConfig> makeNoZSROBoardConfig(uint16_t gbtUniqueId = 0xFFFF);

} // namespace mid
} // namespace o2

#endif /* O2_MID_ROBOARDCONFIGHANDLER_H */
