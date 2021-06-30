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

/// \file   MIDRaw/ColumnDataToLocalBoard.h
/// \brief  Converter from ColumnData to raw local boards
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   20 April 2020
#ifndef O2_MID_COLUMNDATATOLOCALBOARD_H
#define O2_MID_COLUMNDATATOLOCALBOARD_H

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <gsl/gsl>
#include "DataFormatsMID/ColumnData.h"
#include "MIDBase/Mapping.h"
#include "MIDRaw/CrateMapper.h"
#include "DataFormatsMID/ROBoard.h"

namespace o2
{
namespace mid
{
class ColumnDataToLocalBoard
{
 public:
  void process(gsl::span<const ColumnData> data);
  /// Gets the output data per GBT link
  const std::unordered_map<uint16_t, std::vector<ROBoard>> getData() { return mGBTMap; }
  /// Sets debug mode
  void setDebugMode(bool debugMode = true) { mDebugMode = debugMode; }

 private:
  bool keepBoard(const ROBoard& loc) const;
  std::unordered_map<uint8_t, ROBoard> mLocalBoardsMap{};       /// Map of data per board
  std::unordered_map<uint16_t, std::vector<ROBoard>> mGBTMap{}; /// Map of data per GBT link
  CrateMapper mCrateMapper{};                                   /// Crate mapper
  Mapping mMapping{};                                           /// Segmentation
  bool mDebugMode{false};                                       /// Debug mode (no zero suppression)
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_COLUMNDATATOLOCALBOARD_H */
