// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/ElectronicsDelay.cxx
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
  /// Output streamer for ElectronicsDelay
  os << "calibToFET: " << delay.calibToFET << "\n";
  os << "BCToLocal: " << delay.BCToLocal << "\n";
  os << "regToLocal: " << delay.regToLocal << "\n";
  return os;
}

ElectronicsDelay readElectronicsDelay(const char* filename)
{
  /// Reads the electronic delays from file
  ElectronicsDelay electronicsDelay;
  std::ifstream inFile(filename);
  if (inFile.is_open()) {
    std::string line;
    while (std::getline(inFile, line)) {
      line.erase(std::remove_if(line.begin(), line.end(), [](unsigned char x) { return std::isspace(x); }), line.end());
      auto pos = line.find(":");
      if (pos != std::string::npos) {
        std::string key = line.substr(0, pos);
        uint16_t val = std::atoi(line.substr(pos + 1).c_str());
        if (key == "calibToFET") {
          electronicsDelay.calibToFET = val;
        } else if (key == "BCToLocal") {
          electronicsDelay.BCToLocal = val;
        } else if (key == "regToLocal") {
          electronicsDelay.regToLocal = val;
        }
      }
    }
  } else {
    std::cout << "Error: cannot open file " << filename << std::endl;
  }
  return electronicsDelay;
}

} // namespace mid
} // namespace o2
