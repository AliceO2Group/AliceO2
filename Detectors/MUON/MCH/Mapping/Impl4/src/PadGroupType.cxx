// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @author  Laurent Aphecetche

#include "PadGroupType.h"
#include "boost/format.hpp"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <iterator>

namespace o2
{
namespace mch
{
namespace mapping
{
namespace impl4
{

namespace
{
int extent(const std::vector<int>& v)
{
  auto result = std::minmax_element(begin(v), end(v));
  return 1 + *result.second - *result.first;
}
} // namespace

std::vector<int> validIndices(const std::vector<int>& ids)
{
  std::vector<int> v;
  for (auto i = 0; i < ids.size(); i++) {
    if (ids[i] >= 0) {
      v.push_back(i);
    }
  }
  return v;
}

PadGroupType::PadGroupType(int nofPadsX, int nofPadsY, std::vector<int> ids)
  : mFastId{std::move(ids)},
    mFastIndices{validIndices(mFastId)},
    mNofPads{static_cast<int>(std::count_if(begin(mFastId), end(mFastId),
                                            [](int i) { return i >= 0; }))},
    mNofPadsX{nofPadsX},
    mNofPadsY{nofPadsY}
{
}

int PadGroupType::id(int index) const
{
  if (index >= 0 && index < mFastId.size()) {
    return mFastId[index];
  }
  return -1;
}

bool PadGroupType::hasPadById(int id) const
{
  return id != -1 &&
         std::find(begin(mFastId), end(mFastId), id) != end(mFastId);
}

void dump(std::ostream& os, std::string msg, const std::vector<int>& v)
{
  os << boost::format("%4s ") % msg;
  for (auto value : v) {
    os << boost::format("%2d ") % value;
  }
  os << "\n";
}

std::ostream& operator<<(std::ostream& os, const PadGroupType& pgt)
{
  os << "n=" << pgt.getNofPads() << " nx=" << pgt.getNofPadsX()
     << " ny=" << pgt.getNofPadsY() << "\n";
  dump(os, "index", pgt.mFastId);
  return os;
}

} // namespace impl4
} // namespace mapping
} // namespace mch

} // namespace o2
