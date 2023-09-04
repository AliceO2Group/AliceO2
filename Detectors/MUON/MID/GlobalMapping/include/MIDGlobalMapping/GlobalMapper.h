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

/// \file   MIDGlobalMapping/GlobalMapper.h
/// \brief  Global mapper for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 April 2023

#ifndef O2_MID_GLOBALMAPPER_H
#define O2_MID_GLOBALMAPPER_H

#include <array>
#include <map>
#include <vector>
#include "MIDBase/Mapping.h"
#include "MIDRaw/CrateMapper.h"
#include "MIDGlobalMapping/ExtendedMappingInfo.h"

namespace o2
{
namespace mid
{
/// Global mapper for MID
class GlobalMapper
{
 public:
  /// @brief Build the strips info
  /// @return Map with strip unique ID and strip info
  std::vector<ExtendedMappingInfo> buildStripsInfo() const;

  /// @brief Build the geometry for the Detection elements
  /// @return Map with detection element ID and vertexes
  std::map<int, std::vector<std::pair<int, int>>> buildDEGeom() const;

  /// @brief Returns the mapping
  /// @return Const reference to mapping object
  const Mapping& getMapping() const { return mMapping; }

  /// @brief Sets the scale factor for geometrical positions and sizes
  /// @param scaleFactor integer scale factor
  void setScaleFactor(int scaleFactor) { mScaleFactor = scaleFactor; }

 private:
  /// @brief Gets the strip geometrical information
  /// @param deId Detection element ID
  /// @param columnId Column ID
  /// @param lineId Line ID
  /// @param stripId Strip ID
  /// @param cathode Bending (0) or non-bending (1) plane
  /// @return Array with xpos, ypos, xwidth, ywidth
  std::array<int, 4> getStripGeom(int deId, int columnId, int lineId, int stripId, int cathode) const;

  /// @brief Gets the extended information of for the strip
  /// @param deId Detection element ID
  /// @param columnId Column ID
  /// @param lineId Line ID
  /// @param stripId Strip ID
  /// @param cathode Bending (0) or non-bending (1) plane
  /// @param idx Strip index
  /// @return Struct containing the extended information for the strip
  ExtendedMappingInfo buildExtendedInfo(int deId, int columnId, int lineId, int stripId, int cathode) const;

  Mapping mMapping;         ///! Mapping
  CrateMapper mCrateMapper; ///! Crate mapper
  int mScaleFactor = 1;     ///! Scale factor
};

/// @brief Gets the unique strip ID
/// @param deId Detection element ID
/// @param columnId Column ID
/// @param lineId Line ID
/// @param stripId Strip ID
/// @param cathode Bending (0) or non-bending (1) plane
/// @return Unique strip ID
int getStripId(int deId, int columnId, int lineId, int stripId, int cathode);

} // namespace mid
} // namespace o2
#endif