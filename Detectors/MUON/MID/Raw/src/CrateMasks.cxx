// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/CrateMasks.cxx
/// \brief  MID crate masks
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 March 2020

#include "MIDRaw/CrateMasks.h"

#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

namespace o2
{
namespace mid
{

CrateMasks::CrateMasks() : mActiveBoards()
{
  /// Default constructor
  for (uint16_t ioffset = 0; ioffset < crateparams::sNGBTs; ioffset += crateparams::sNGBTsPerSide) {
    // Crate 1
    mActiveBoards[0 + ioffset] = 0xFF;
    mActiveBoards[1 + ioffset] = 0xFF;

    // Crate 2
    mActiveBoards[2 + ioffset] = 0xFF;
    mActiveBoards[3 + ioffset] = 0x7F;

    // Crate 2-3
    mActiveBoards[4 + ioffset] = 0x7F;
    mActiveBoards[5 + ioffset] = 0x7F;

    // Crate 3
    mActiveBoards[6 + ioffset] = 0xFF;
    mActiveBoards[7 + ioffset] = 0x7F;

    // Crate 4
    mActiveBoards[8 + ioffset] = 0xFF;
    mActiveBoards[9 + ioffset] = 0xFF;

    // Crate 5
    mActiveBoards[10 + ioffset] = 0xFF;
    mActiveBoards[11 + ioffset] = 0xFF;

    // Crate 6
    mActiveBoards[12 + ioffset] = 0xFF;
    mActiveBoards[13 + ioffset] = 0xFF;

    // Crate 7
    mActiveBoards[14 + ioffset] = 0xFF;
    mActiveBoards[15 + ioffset] = 0x1;
  }
}

CrateMasks::CrateMasks(const char* filename) : mActiveBoards()
{
  /// Construct from file
  load(filename);
}

bool CrateMasks::load(const char* filename)
{
  /// Loads the masks from a configuration file
  /// The file is in the form:
  /// feeId mask
  /// with one line per link
  /// The mask is at most 8 bits, since each GBT link reads at most 8 local boards
  mActiveBoards.fill(0);
  std::ifstream inFile(filename);
  if (!inFile.is_open()) {
    return false;
  }
  std::string line, token;
  while (std::getline(inFile, line)) {
    if (std::count(line.begin(), line.end(), ' ') < 1) {
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
    uint8_t mask = static_cast<uint8_t>(std::strtol(token.c_str(), nullptr, 16));
    mActiveBoards[feeId] = mask;
  }
  inFile.close();
  return true;
}

void CrateMasks::write(const char* filename) const
{
  /// Writes the masks to a configuration file
  std::ofstream outFile(filename);
  for (uint16_t igbt = 0; igbt < crateparams::sNGBTs; ++igbt) {
    outFile << igbt << " " << mActiveBoards[igbt] << std::endl;
  }
  outFile.close();
}

} // namespace mid
} // namespace o2
