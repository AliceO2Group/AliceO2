// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/TextControlService.h"
#include "FairMQLogger.h"
#include <string>
#include <regex>
#include <iostream>

namespace o2 {
namespace framework {

// All we do is to printout
void TextControlService::readyToQuit(bool all) {
  if (mOnce == true) {
    return;
  }
  mOnce = true;
  std::cout << "CONTROL_ACTION: READY_TO_QUIT";
  if (all) {
    std::cout << "_ALL";
  } else {
    std::cout << "_ME";
  }
  std::cout << std::endl;
}

bool parseControl(const std::string &s, std::smatch &match) {
  const static std::regex controlRE(".*CONTROL_ACTION: READY_TO_(QUIT)_(ME|ALL)");
  return std::regex_match(s, match, controlRE);
}

} // framework
} // o2
