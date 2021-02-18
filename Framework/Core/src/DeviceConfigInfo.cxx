// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DeviceConfigInfo.h"
#include "Framework/DeviceInfo.h"
#include <cstdlib>
#include <regex>
#include <string_view>

namespace o2::framework
{

// Parses a config entry in the form
//
// [CONFIG] <key>=<vaue> <timestamp> <provenance>
bool DeviceConfigHelper::parseConfig(std::string_view s, ParsedConfigMatch& match)
{
  const char* begin = s.begin();
  const char* end = s.end();
  if (s.size() > 17 && (strncmp("[CONFIG] ", begin + 17, 9) != 0)) {
    return false;
  }
  if (s.size() < 17 + 9) {
    return false;
  }
  match.beginKey = s.data() + 9 + 17;
  match.endKey = (char const*)memchr(match.beginKey, '=', s.size() - 9);
  if (match.endKey == nullptr) {
    return false;
  }
  match.beginValue = match.endKey + 1;
  match.endValue = (char const*)memchr(match.beginValue, ' ', end - match.beginValue);
  if (match.endValue == nullptr) {
    return false;
  }
  char* next = nullptr;
  match.timestamp = strtoll(match.endValue, &next, 10);

  if (!next || *next != ' ') {
    return false;
  }

  match.beginProvenance = next + 1;
  match.endProvenance = end;

  return true;
}

bool DeviceConfigHelper::processConfig(ParsedConfigMatch& match,
                                       DeviceInfo& info)
{
  if (match.beginKey == nullptr || match.endKey == nullptr ||
      match.beginValue == nullptr || match.endValue == nullptr ||
      match.beginProvenance == nullptr || match.endProvenance == nullptr) {
    return false;
  }
  auto keyString = std::string(match.beginKey, match.endKey - match.beginKey);
  auto valueString = std::string(match.beginValue, match.endValue - match.beginValue);
  auto provenanceString = std::string(match.beginProvenance, match.endProvenance - match.beginProvenance);
  std::regex fmatch(R"([ifdb]\[.*\])", std::regex_constants::ECMAScript);
  std::regex nmatch(R"((?:(?!=,)|(?!=\[))[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?(?=,|\]))", std::regex_constants::ECMAScript);
  auto end = std::sregex_iterator();
  auto fmt = std::sregex_iterator(valueString.begin(), valueString.end(), fmatch);
  if (fmt != end) {
    boost::property_tree::ptree branch;
    auto values = std::sregex_iterator(valueString.begin(), valueString.end(), nmatch);
    for (auto v = values; v != end; ++v) {
      boost::property_tree::ptree leaf;
      leaf.put("", v->str());
      branch.push_back(std::make_pair("", leaf));
    }
    info.currentConfig.put_child(keyString, branch);
  } else {
    info.currentConfig.put(keyString, valueString);
  }
  info.currentProvenance.put(keyString, provenanceString);
  return true;
}

} // namespace o2::framework
