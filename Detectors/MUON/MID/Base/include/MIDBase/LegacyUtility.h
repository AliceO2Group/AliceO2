// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDBase/LegacyUtility.h
/// \brief  Utility for MID detection element IDs
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   31 August 2017
#ifndef O2_MID_LegacyUtility_H
#define O2_MID_LegacyUtility_H

#include <cstdint>
#include <string>
#include <vector>
#include "DataFormatsMID/ColumnData.h"

namespace o2
{
namespace mid
{
/// Set of utilities allowing to pass from the old Run2 formats to the new one
/// and viceversa.
/// These utilities are meant for testing, allowing one to compare
/// with the AliRoot input/output on existing data.
/// It is of course doomed to disappear with time.
struct LegacyUtility {
  static bool assertLegacyDeId(int detElemId);
  static bool assertDeId(int deId);
  static int convertFromLegacyDeId(int detElemId);
  static int convertToLegacyDeId(int deId);
  static std::string legacyDeIdName(int detElemId);
  static std::string deIdName(int index);
  static uint32_t encodeDigit(int detElemId, int boardId, int channel, int cathode);
  static void decodeDigit(uint32_t uniqueId, int& detElemId, int& boardId, int& strip, int& cathode);
  static std::vector<ColumnData> digitsToPattern(std::vector<uint32_t> digits);
  static void boardToPattern(int boardId, int detElemId, int cathode, int& deId, int& column, int& line);
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_LegacyUtility_H */
