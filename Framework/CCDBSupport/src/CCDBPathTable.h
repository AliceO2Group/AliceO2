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
#ifndef O2_FRAMEWORK_CCDBPATHTABLE_H_
#define O2_FRAMEWORK_CCDBPATHTABLE_H_

#include <Framework/Logger.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace o2::framework
{
// A CCDB path may be declared either as a plain path, or as a mapping from uniformity
// value to path: "lo-hi=path;lo-hi=path;fallback". Ranges are inclusive and either bound
// may be omitted ("-hi=path", "lo-=path"). An entry without '=' is an explicit fallback;
// without one, a value matching no range is an error rather than a silent guess.
// The mapping is data, carried in the schema metadata, so the fetcher needs no code from
// the task that declared the column.
struct PathTable {
  struct Range {
    int64_t lo;
    int64_t hi;
    std::string path;
  };
  std::vector<Range> ranges;
  std::string fallback;
  bool hasFallback = false;

  static PathTable parse(std::string const& spec)
  {
    PathTable table;
    if (spec.find('=') == std::string::npos) { // plain path, the common case
      table.fallback = spec;
      table.hasFallback = true;
      return table;
    }
    size_t pos = 0;
    while (pos <= spec.size()) {
      auto end = spec.find(';', pos);
      auto entry = spec.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
      pos = (end == std::string::npos) ? spec.size() + 1 : end + 1;
      if (entry.empty()) {
        continue;
      }
      auto eq = entry.find('=');
      if (eq == std::string::npos) {
        table.fallback = entry;
        table.hasFallback = true;
        continue;
      }
      auto bounds = entry.substr(0, eq);
      auto dash = bounds.find('-');
      if (dash == std::string::npos) {
        LOGP(fatal, R"(Malformed CCDB path mapping "{}": expected "lo-hi=path")", entry);
      }
      auto loStr = bounds.substr(0, dash);
      auto hiStr = bounds.substr(dash + 1);
      table.ranges.push_back({loStr.empty() ? std::numeric_limits<int64_t>::min() : std::stoll(loStr),
                              hiStr.empty() ? std::numeric_limits<int64_t>::max() : std::stoll(hiStr),
                              entry.substr(eq + 1)});
    }
    return table;
  }

  std::string const& resolve(int64_t key, std::string const& column) const
  {
    for (auto const& range : ranges) {
      if (key >= range.lo && key <= range.hi) {
        return range.path;
      }
    }
    if (!hasFallback) {
      LOGP(fatal, R"(No CCDB path declared for {} at uniformity value {}; the declared mapping covers no such value and has no fallback entry)",
           column, key);
    }
    return fallback;
  }
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_CCDBPATHTABLE_H_
