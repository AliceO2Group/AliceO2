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

#include <iostream>
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
  /// Constructor from stream
  ROBoardConfigHandler(std::istream& in);
  /// Constructor from list of local board configuration
  ROBoardConfigHandler(const std::vector<ROBoardConfig>& configurations);
  /// Default destructor
  ~ROBoardConfigHandler() = default;

  /// Returns the configuration for the local board
  /// \param uniqueLocId Unique local board ID
  /// \return Readout Board configuration
  const ROBoardConfig getConfig(uint8_t uniqueLocId) const;

  /// Returns the configuration map
  const std::unordered_map<uint8_t, ROBoardConfig> getConfigMap() const { return mROBoardConfigs; }

  /// Sets the local board configurations from a vector
  /// \param configurations List of local board configurations
  void set(const std::vector<ROBoardConfig>& configurations);

  /// Updates the mask values
  /// \param masks New masks
  void updateMasks(const std::vector<ROBoard>& masks);

  /// Writes the configuration to file
  /// \param filename Output file path
  void write(const char* filename) const;

  /// Streams the configuration
  /// \param out Output stream
  void write(std::ostream& out) const;

 private:
  /// Loads the board from a configuration file
  /// The file is in the form:
  /// locId status maskX1Y1 maskX2Y2 maskX3Y3 maskX4Y4
  /// with one line per local board
  /// \param filename Input filename
  bool load(const char* filename);

  /// Loads the board from a configuration stream
  /// \param in Input stream
  void load(std::istream& in);

  std::unordered_map<uint8_t, ROBoardConfig> mROBoardConfigs; /// Vector of local board configuration
};

/// Creates the default local board configurations
/// \param gbtUniqueId GBT unique ID
/// \return Vector of Readout boards configuration
std::vector<ROBoardConfig> makeDefaultROBoardConfig(uint16_t gbtUniqueId = 0xFFFF);
/// Creates a local board configuration where no zero suppression is required
/// \param gbtUniqueId GBT unique ID
/// \return Vector of Readout boards configuration with no zero suppression
std::vector<ROBoardConfig> makeNoZSROBoardConfig(uint16_t gbtUniqueId = 0xFFFF);
/// Creates a local board configuration with zero suppression
/// \param gbtUniqueId GBT unique ID
/// \return Vector of Readout boards configuration with zero suppression
std::vector<ROBoardConfig> makeZSROBoardConfig(uint16_t gbtUniqueId = 0xFFFF);
} // namespace mid
} // namespace o2

#endif /* O2_MID_ROBOARDCONFIGHANDLER_H */
