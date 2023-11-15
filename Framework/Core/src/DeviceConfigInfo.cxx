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

#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include "Framework/DeviceConfigInfo.h"
#include "Framework/DeviceInfo.h"
#include <cstdlib>
#include <string_view>
#include <boost/property_tree/json_parser.hpp>

namespace o2::framework
{

// Parses a config entry in the form
//
// [CONFIG] <key>=<vaue> <timestamp> <provenance>
//
// Now with backtracking to parse timestamp and provenance
bool DeviceConfigHelper::parseConfig(std::string_view s, ParsedConfigMatch& match)
{
  const char* cur = s.data();
  const char* next = cur;
  enum struct ParserState {
    IN_PREAMBLE,
    IN_KEY,
    IN_VALUE,
    IN_TIMESTAMP,
    IN_PROVENANCE,
    IN_ERROR,
    DONE
  };
  char* err = nullptr;
  // We need to keep track of the last and last but one space
  // to be able to parse the timestamp and tags.
  char const* lastSpace = nullptr;
  char const* previousLastSpace = nullptr;
  ParserState state = ParserState::IN_PREAMBLE;

  while (true) {
    auto previousState = state;
    state = ParserState::IN_ERROR;
    err = nullptr;
    switch (previousState) {
      case ParserState::IN_PREAMBLE:
        if (s.data() + s.size() - cur < 9) {
        } else if (strncmp("[CONFIG] ", cur, 9) == 0) {
          next = cur + 8;
          state = ParserState::IN_KEY;
        }
        break;
      case ParserState::IN_KEY:
        next = strpbrk(cur, "= ");
        // Invalid key
        if (next == nullptr || *next == ' ' || (next == cur)) {
        } else if (*next == '=') {
          match.beginKey = cur;
          match.endKey = next;
          match.beginValue = next + 1;
          state = ParserState::IN_VALUE;
        }
        break;
      case ParserState::IN_VALUE:
        next = (char*)memchr(cur, ' ', s.data() + s.size() - cur);
        if (next == nullptr) {
          if (previousLastSpace == nullptr || lastSpace == nullptr) {
            // We need at least two spaces to parse the timestamp and
            // the provenance.
            break;
          }
          match.endValue = previousLastSpace;
          next = previousLastSpace;
          state = ParserState::IN_TIMESTAMP;
        } else {
          previousLastSpace = lastSpace;
          lastSpace = next;
          state = ParserState::IN_VALUE;
        }
        break;
      case ParserState::IN_TIMESTAMP:
        match.timestamp = strtoll(cur, &err, 10);
        next = err;
        if (*next == ' ') {
          state = ParserState::IN_PROVENANCE;
        }
        break;
      case ParserState::IN_PROVENANCE:
        match.beginProvenance = cur;
        next = (char*)memchr(cur, '\n', s.data() + s.size() - cur);
        if (next != nullptr) {
          match.endProvenance = next;
          state = ParserState::DONE;
        } else {
          match.endProvenance = s.data() + s.size();
          state = ParserState::DONE;
        }
        break;
      case ParserState::IN_ERROR:
        return false;
      case ParserState::DONE:
        return true;
    }
    cur = next + 1;
  }
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
  boost::property_tree::ptree branch;
  std::stringstream ss{valueString};
  try {
    boost::property_tree::json_parser::read_json(ss, branch);
    info.currentConfig.put_child(keyString, branch);
  } catch (boost::exception&) {
    // in case it is not a tree but a single value
    info.currentConfig.put(keyString, valueString);
  }

  info.currentProvenance.put(keyString, provenanceString);
  return true;
}

} // namespace o2::framework
