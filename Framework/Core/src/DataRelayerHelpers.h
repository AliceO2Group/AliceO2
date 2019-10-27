// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DATARELAYERHELPERS_H_
#define O2_FRAMEWORK_DATARELAYERHELPERS_H_

#include "Framework/InputRoute.h"
#include <vector>

namespace o2::framework
{

struct DataRelayerHelpers {
  /// Calculate how many input routes there are, doublecounting different
  /// timeslices.
  static std::vector<size_t> createDistinctRouteIndex(std::vector<InputRoute> const&);
  /// This converts from InputRoute to the associated DataDescriptorMatcher.
  static std::vector<data_matcher::DataDescriptorMatcher> createInputMatchers(std::vector<InputRoute> const&);
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATARELAYERHELPERS_H_
