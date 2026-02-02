// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <string>

namespace o2::framework
{
/// Convert a CamelCase task struct name to snake-case task name
std::string type_to_task_name(std::string_view const& camelCase)
{
  std::string result;
  result.reserve(camelCase.size() * 2 + 2);

  // The first character is always -.
  result += "-";
  result += static_cast<char>(std::tolower(camelCase[0]));

  for (auto it = camelCase.begin() + 1; it != camelCase.end(); ++it) {
    if (std::isupper(*it) && *(it - 1) != '-') {
      result += '-';
    }
    result += static_cast<char>(std::tolower(*it));
  }
  // Post-process to consolidate common ALICE abbreviations
  // Process backwards to handle patterns correctly
  static const struct {
    std::string_view pattern;
    std::string_view replacement;
  } abbreviations[] = {
    {"-e-m-c-a-l", "-emcal"},
    {"-e-m-c", "-emc"}
  };

  std::string consolidated;
  consolidated.reserve(result.size());

  for (int i = result.size() - 1; i >= 0;) {
    bool matched = false;

    for (const auto& abbr : abbreviations) {
      int startPos = i - abbr.pattern.size() + 1;
      if (startPos >= 0 && result.compare(startPos, abbr.pattern.size(), abbr.pattern.data()) == 0) {
        consolidated.insert(0, abbr.replacement);
        i = startPos - 1;
        matched = true;
        break;
      }
    }

    if (!matched) {
      consolidated.insert(0, 1, result[i]);
      --i;
    }
  }
  if (consolidated[0] == '-') {
    return std::string(consolidated.data() + 1);
  }

  return consolidated;
}
} // namespace o2::framework
