// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Base/src/LegacyUtility.cxx
/// \brief  Utility for MID detection element IDs
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   31 August 2017
#include "MIDBase/LegacyUtility.h"
#include <iostream>

namespace o2
{
namespace mid
{
//______________________________________________________________________________
bool LegacyUtility::assertLegacyDeId(int detElemId)
{
  /// Checks if the detection element id is valid (Run2 format)
  int chamberId = detElemId / 100;
  int rpcId = detElemId % 100;
  if (chamberId < 11 || chamberId > 14 || rpcId > 17) {
    std::cerr << "Invalid detElemId: " << detElemId << std::endl;
    return false;
  }
  return true;
}

//______________________________________________________________________________
bool LegacyUtility::assertDeId(int deId)
{
  /// Checks if the detection element id is valid
  if (deId > 71) {
    std::cerr << "Invalid deId: " << deId << std::endl;
    return false;
  }
  return true;
}

//______________________________________________________________________________
int LegacyUtility::convertFromLegacyDeId(int detElemId)
{
  /// Converts the detection element ID (Run2 format)
  /// into the new ID (Run3 format)
  if (assertLegacyDeId(detElemId)) {
    int ich = (detElemId / 100 - 11);
    int irpc = (detElemId % 100 + 4) % 18;
    if (irpc >= 9) {
      irpc = 17 - irpc;
      ich += 4;
    }
    return ich * 9 + irpc;
  }

  return 0xFF;
}

//______________________________________________________________________________
int LegacyUtility::convertToLegacyDeId(int deId)
{
  /// Converts the detection element ID into the Run2 format
  if (assertDeId(deId)) {
    int ich = (deId % 36) / 9;
    int irpc = 0;
    if (deId / 36 == 0) {
      irpc = (deId % 9 + 14) % 18;
    } else {
      irpc = 13 - (deId % 9);
    }
    return (ich + 11) * 100 + irpc;
  }

  return 0xFFFF;
}

//______________________________________________________________________________
std::string LegacyUtility::deIdName(int deId)
{
  /// Returns the name of the detection element
  if (assertDeId(deId)) {
    int chId = deId / 18;
    int stId = 1 + chId / 2;
    int planeId = 1 + chId % 2;
    std::string str = "MT";
    str += stId;
    str += planeId;
    str += (deId / 9 == 0) ? "In" : "Out";
    str += deId % 18;
    return str;
  }
  return "";
}

//______________________________________________________________________________
std::string LegacyUtility::legacyDeIdName(int detElemId)
{
  /// Returns the name of the detection element (expressed in the Run2 format)
  if (assertLegacyDeId(detElemId)) {
    return deIdName(convertFromLegacyDeId(detElemId));
  }
  return "";
}

//______________________________________________________________________________
uint32_t LegacyUtility::encodeDigit(int detElemId, int boardId, int strip, int cathode)
{
  /// Encodes the digit into a uniqueId in the Run2 format
  uint32_t uniqueId = 0;
  uniqueId |= detElemId;
  uniqueId |= (boardId << 12);
  uniqueId |= (strip << 24);
  uniqueId |= (cathode << 30);
  return uniqueId;
}

//______________________________________________________________________________
void LegacyUtility::decodeDigit(uint32_t uniqueId, int& detElemId, int& boardId, int& strip, int& cathode)
{
  /// Decodes the digit given as a uniqueId in the Run2 format
  detElemId = uniqueId & 0xFFF;
  boardId = (uniqueId & 0xFFF000) >> 12;
  strip = (uniqueId & 0x3F000000) >> 24;
  cathode = (uniqueId & 0x40000000) >> 30;
}

//______________________________________________________________________________
void LegacyUtility::boardToPattern(int boardId, int detElemId, int cathode, int& deId, int& column, int& line)
{
  /// Converts old Run2 local board Id into the new format
  deId = convertFromLegacyDeId(detElemId);
  int iboard = (boardId - 1) % 117;
  int halfBoardId = iboard + 1;
  int endBoard[7] = { 16, 38, 60, 76, 92, 108, 117 };
  for (int icol = 0; icol < 7; ++icol) {
    if (halfBoardId > endBoard[icol]) {
      continue;
    }
    column = icol;
    break;
  }
  line = 0;

  if (cathode == 1) {
    return;
  }

  std::vector<int> lines[3];
  lines[0] = { 3,  19,  41, 63, 79, 95, 5,  21,  43, 65, 81, 97, 7,  23,  45, 67, 83, 99, 27, 49, 69,
               85, 101, 9,  31, 53, 71, 87, 103, 13, 35, 57, 73, 89, 105, 15, 37, 59, 75, 91, 107 };
  lines[1] = { 8, 24, 46, 28, 50, 10, 32, 54 };
  lines[2] = { 25, 47, 29, 51, 11, 33, 55 };
  for (int il = 0; il < 3; ++il) {
    for (auto& val : lines[il]) {
      if (halfBoardId == val) {
        line = il + 1;
        return;
      }
    }
  }
}

//______________________________________________________________________________
std::vector<ColumnData> LegacyUtility::digitsToPattern(std::vector<uint32_t> digits)
{
  /// Converts digits in the old Run2 format to StripPattern
  int detElemId, boardId, channel, cathode, icolumn, iline;
  int deId;
  std::vector<ColumnData> columns;
  for (auto uniqueId : digits) {
    decodeDigit(uniqueId, detElemId, boardId, channel, cathode);
    boardToPattern(boardId, detElemId, cathode, deId, icolumn, iline);
    ColumnData* currentColumn = nullptr;
    for (auto& column : columns) {
      if (column.deId != deId) {
        continue;
      }
      if (column.columnId != icolumn) {
        continue;
      }
      currentColumn = &column;
      break;
    }
    if (!currentColumn) {
      columns.emplace_back(ColumnData());
      currentColumn = &columns.back();
      currentColumn->deId = deId;
      currentColumn->columnId = icolumn;
    }
    uint16_t pattern =
      (cathode == 0) ? currentColumn->patterns.getBendPattern(iline) : currentColumn->patterns.getNonBendPattern();
    pattern |= (1 << channel);
    if (cathode == 0) {
      currentColumn->patterns.setBendPattern(pattern, iline);
    } else {
      currentColumn->patterns.setNonBendPattern(pattern);
    }
  }
  return columns;
}

} // namespace mid
} // namespace o2
