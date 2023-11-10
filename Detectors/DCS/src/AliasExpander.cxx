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

#include "DetectorsDCS/AliasExpander.h"
#include <algorithm>
#include <fmt/format.h>
#include <sstream>

namespace
{

std::vector<std::string> splitString(const std::string& src, char delim)
{
  std::stringstream ss(src);
  std::string token;
  std::vector<std::string> tokens;

  while (std::getline(ss, token, delim)) {
    if (!token.empty()) {
      tokens.push_back(std::move(token));
    }
  }

  return tokens;
}

std::vector<std::string> extractList(const std::string& slist)
{
  auto dots = slist.find(",");
  if (dots == std::string::npos) {
    return {};
  }
  return splitString(slist, ',');
}

std::vector<std::string> extractRange(std::string range)
{
  auto dots = range.find("..");
  if (dots == std::string::npos) {
    return extractList(range);
  }

  auto braceStart = range.find("{");
  auto braceEnd = range.find("}");

  if (
    (braceStart != std::string::npos &&
     braceEnd == std::string::npos) ||
    (braceStart == std::string::npos &&
     braceEnd != std::string::npos)) {
    // incomplete custom pattern
    return {};
  }

  std::string intFormat;
  std::string sa, sb;

  if (braceStart != std::string::npos &&
      braceEnd != std::string::npos) {
    intFormat = range.substr(braceStart, braceEnd - braceStart + 1);
    range.erase(braceStart, braceEnd);
    dots = range.find("..");
    sa = range.substr(0, dots);
    sb = range.substr(dots + 2);
  } else {
    sa = range.substr(0, dots);
    sb = range.substr(dots + 2);
    auto size = std::max(sa.size(), sb.size());
    intFormat = "{:" + fmt::format("0{}d", size) + "}";
  }

  auto a = std::stoi(sa);
  auto b = std::stoi(sb);
  std::vector<std::string> result;

  for (auto i = a; i <= b; i++) {
    auto substituted = fmt::format(fmt::runtime(intFormat), i);
    result.push_back(substituted);
  }
  return result;
}
} // namespace

namespace o2::dcs
{
std::vector<std::string> expandAlias(const std::string& pattern)
{
  auto leftBracket = pattern.find("[");
  auto rightBracket = pattern.find("]");

  // no bracket at all -> return pattern simply
  if (leftBracket == std::string::npos && rightBracket == std::string::npos) {
    return {pattern};
  }

  // no matching bracket -> wrong pattern -> return nothing
  if ((leftBracket == std::string::npos &&
       rightBracket != std::string::npos) ||
      (leftBracket != std::string::npos &&
       rightBracket == std::string::npos)) {
    return {};
  }
  auto rangeStr = pattern.substr(leftBracket + 1, rightBracket - leftBracket - 1);

  auto range = extractRange(rangeStr);

  // incorrect range -> return nothing
  if (range.empty()) {
    return {};
  }

  auto newPattern = pattern.substr(0, leftBracket) +
                    "{:s}" +
                    pattern.substr(rightBracket + 1);

  std::vector<std::string> result;

  for (auto r : range) {
    auto substituted = fmt::format(fmt::runtime(newPattern), r);
    result.emplace_back(substituted);
  }

  return o2::dcs::expandAliases(result);
}

std::vector<std::string> expandAliases(const std::vector<std::string>& patternedAliases)
{
  std::vector<std::string> result;

  for (auto a : patternedAliases) {
    auto e = expandAlias(a);
    result.insert(result.end(), e.begin(), e.end());
  }
  // sort to get a predictable result
  std::sort(result.begin(), result.end());

  return result;
}
} // namespace o2::dcs
