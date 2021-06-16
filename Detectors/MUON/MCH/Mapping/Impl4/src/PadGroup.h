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

///
/// @author  Laurent Aphecetche

#ifndef O2_MCH_MAPPING_impl4_PADGROUP_H
#define O2_MCH_MAPPING_impl4_PADGROUP_H

#include <ostream>

namespace o2
{
namespace mch
{
namespace mapping
{
namespace impl4
{

struct PadGroup {
  friend std::ostream& operator<<(std::ostream& os, const PadGroup& group)
  {
    os << "mFECId: " << group.mFECId
       << " mPadGroupTypeId: " << group.mPadGroupTypeId
       << " mPadSizeId: " << group.mPadSizeId << " mX: " << group.mX
       << " mY: " << group.mY;
    return os;
  }

  int mFECId;
  int mPadGroupTypeId;
  int mPadSizeId;
  double mX;
  double mY;
};

} // namespace impl4
} // namespace mapping
} // namespace mch
} // namespace o2
#endif
