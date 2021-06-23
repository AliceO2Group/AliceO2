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
#ifndef FRAMEWORK_FORWARDROUTE_H
#define FRAMEWORK_FORWARDROUTE_H

#include "Framework/InputSpec.h"
#include <cstddef>
#include <string>

namespace o2
{
namespace framework
{

/// This uniquely identifies a route to be forwarded by the device if
/// the InputSpec @a matcher matches an input which should also go to
/// @a channel
struct ForwardRoute {
  size_t timeslice;
  size_t maxTimeslices;
  InputSpec matcher;
  std::string channel;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_FORWARDROUTE_H
