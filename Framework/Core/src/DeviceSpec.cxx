// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "../src/WorkflowHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ChannelMatching.h"
#include "Framework/DeviceControl.h"
#include <vector>
#include <algorithm>
#include <unordered_set>

using namespace o2::framework;

namespace o2
{
namespace framework
{

using LogicalChannelsMap = std::map<LogicalChannelRange, size_t>;

// This calculates the distance between two strings. See:
//
// https://en.wikipedia.org/wiki/Levenshtein_distance
//
// For the full description
size_t levenshteinDistance(const char* s, int len_s, const char* t, int len_t)
{
  size_t cost;

  /* base case: empty strings */
  if (len_s == 0)
    return len_t;
  if (len_t == 0)
    return len_s;

  /* test if last characters of the strings match */
  if (s[len_s - 1] == t[len_t - 1])
    cost = 0;
  else
    cost = 1;

  return std::min(std::min(levenshteinDistance(s, len_s - 1, t, len_t) + 1,
                           levenshteinDistance(s, len_s, t, len_t - 1) + 1),
                  levenshteinDistance(s, len_s - 1, t, len_t - 1) + cost);
}

std::string findBestCandidate(const std::string& candidate, const LogicalChannelsMap& map)
{
  std::string result;
  size_t score = -1;
  for (const auto& pair : map) {
    auto newScore = levenshteinDistance(candidate.c_str(), candidate.size(),
                                        pair.first.name.c_str(), pair.first.name.size());
    if (newScore < score) {
      result = pair.first.name;
    }
  }
  return result;
}

} // namespace framework
} // namespace o2
