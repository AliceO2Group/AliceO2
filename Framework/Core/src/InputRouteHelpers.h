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

#ifndef O2_FRAMEWORK_INPUTROUTESHELPERS_H_
#define O2_FRAMEWORK_INPUTROUTESHELPERS_H_

#include "Framework/InputRoute.h"
#include <vector>

namespace o2::framework
{

struct InputRouteHelpers {
  /// @return the maximum number of lanes (i.e. max timeslice) which
  /// is associated to a set of input @a routes. This is needed to
  /// make sure we can reserve a set of slots to the given lane.
  static size_t maxLanes(std::vector<InputRoute> const& routes);
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_INPUTROUTESHELPERS_H_
