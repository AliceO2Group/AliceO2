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
#ifndef O2_FRAMEWORK_OUTPUTROUTE_H_
#define O2_FRAMEWORK_OUTPUTROUTE_H_

#include "Framework/OutputSpec.h"
#include <cstddef>
#include <string>

namespace o2::framework
{

struct SendingPolicy;

// This uniquely identifies a route out of the device if
// the OutputSpec @a matcher and @a timeslice match.
struct OutputRoute {
  size_t timeslice;
  size_t maxTimeslices;
  OutputSpec matcher;
  std::string channel;
  // The policy to use to send to on this route.
  SendingPolicy const* policy;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_OUTPUTROUTE_H_
