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

/// \file   MID/Raw/src/ElectronicsDelay.cxx
/// \brief  Delay parameters for MID electronics
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   27 July 2020

#include "MIDRaw/ElectronicsDelay.h"

#include <algorithm>
#include <fstream>
#include <string>

namespace o2
{
namespace mid
{

std::ostream& operator<<(std::ostream& os, const ElectronicsDelay& delay)
{
  os << "calibToFET: " << delay.calibToFET << "\n";
  os << "localToBC: " << delay.localToBC << "\n";
  os << "localToReg: " << delay.localToReg << "\n";
  return os;
}

ElectronicsDelay readElectronicsDelay(const char* filename)
{
  ElectronicsDelay electronicsDelay;
  std::ifstream inFile(filename);
  if (inFile.is_open()) {
    std::string line;
    while (std::getline(inFile, line)) {
      line.erase(std::remove_if(line.begin(), line.end(), [](unsigned char x) { return std::isspace(x); }), line.end());
      auto pos = line.find(":");
      if (pos != std::string::npos) {
        std::string key = line.substr(0, pos);
        int16_t val = std::atoi(line.substr(pos + 1).c_str());
        if (key == "calibToFET") {
          electronicsDelay.calibToFET = val;
        } else if (key == "localToBC") {
          electronicsDelay.localToBC = val;
        } else if (key == "localToReg") {
          electronicsDelay.localToReg = val;
        }
      }
    }
  } else {
    std::cout << "Error: cannot open file " << filename << std::endl;
  }
  return electronicsDelay;
}

void applyElectronicsDelay(uint32_t& orbit, uint16_t& bc, int16_t delay, uint16_t maxBunches)
{
  int16_t val = static_cast<int16_t>(bc) + delay;
  int16_t resetPeriod = static_cast<int16_t>(maxBunches);
  if (val < 0) {
    // If corrected clock is smaller than 0 it means that the local clock was reset
    // This event therefore belongs to the previous orbit.
    // We therefore add the value of the last BC (+1 to account for the reset)
    // and we decrease the orbit by 1.
    --orbit;
    val += resetPeriod;
  } else if (val >= resetPeriod) {
    // If the corrected clock is larger than the maximum clock (corresponding to the reset)
    // it means that this event belongs to the next orbit
    ++orbit;
    val -= resetPeriod;
  }
  // The previous line ensure that 0<val<maxBunches, so we can safely convert the int in unit16_t
  bc = static_cast<uint16_t>(val);
}

} // namespace mid
} // namespace o2
