// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DEVICEINFO_H
#define FRAMEWORK_DEVICEINFO_H

#include "Framework/Variant.h"

#include <vector>
#include <string>
#include <cstddef>
// For pid_t 
#include <unistd.h>
#include <array>

namespace o2 {
namespace framework {

struct DeviceInfo {
  pid_t pid;
  size_t historyPos;
  size_t historySize;
  std::vector<std::string> history;
  std::string unprinted;
  bool active;

};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_DEVICEINFO_H
