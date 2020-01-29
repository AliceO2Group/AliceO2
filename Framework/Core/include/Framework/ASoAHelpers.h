// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_ASOAHELPERS_H_
#define O2_FRAMEWORK_ASOAHELPERS_H_

#include "Framework/ASoA.h"

namespace o2::soa
{

/// @return a vector of pairs with all the possible
/// combinations of the rows of the table T.
/// FIXME: move to coroutines once we have C++20
template <typename T>
std::vector<std::pair<typename T::iterator, typename T::iterator>>
  combinations(T const& table)
{
  std::vector<std::pair<typename T::iterator, typename T::iterator>> result;
  result.reserve((table.size() + 1) * table.size() / 2);
  for (auto t0 = table.begin(); t0 + 1 != table.end(); ++t0) {
    for (auto t1 = t0 + 1; t1 != table.end(); ++t1) {
      result.push_back(std::make_pair(t0, t1));
    }
  }
  return result;
};

} // namespace o2::soa

#endif // O2_FRAMEWORK_ASOAHELPERS_H_
