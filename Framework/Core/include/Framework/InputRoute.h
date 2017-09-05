// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_INPUTROUTE_H
#define FRAMEWORK_INPUTROUTE_H

#include "Framework/InputSpec.h"
#include <cstddef>
#include <string>

namespace o2 {
namespace framework {

/// This uniquely identifies a route to from which data matching @a matcher
/// input spec gets to the device.
struct InputRoute {
  InputSpec matcher;
  std::string sourceChannel;
};

} // framework
} // o2
#endif // FRAMEWORK_INPUTROUTE_H
