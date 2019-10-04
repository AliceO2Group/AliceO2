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
#include "Framework/Logger.h"
#include <string>
#include <string_view>
#include <regex>
#include <iostream>

namespace o2
{
namespace framework
{

// All we do is to printout
void TextControlService::readyToQuit(bool all)
{
  if (mOnce == true) {
    return;
  }
  mOnce = true;
  if (all) {
    LOG(INFO) << "CONTROL_ACTION: READY_TO_QUIT_ALL";
  } else {
    LOG(INFO) << "CONTROL_ACTION: READY_TO_QUIT_ME";
  }
}

bool parseControl(std::string const& s, std::smatch& match)
{
  const static std::regex controlRE(".*CONTROL_ACTION: READY_TO_(QUIT)_(ME|ALL)", std::regex::optimize);
  return std::regex_search(s, match, controlRE);
}

} // namespace framework
} // namespace o2
