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
#ifndef FRAMEWORK_OUTPUTROUTE_H
#define FRAMEWORK_OUTPUTROUTE_H

#include "Framework/OutputSpec.h"
#include <cstddef>
#include <string>

namespace o2
{
namespace framework
{

// This uniquely identifies a route out of the device if
// the OutputSpec @a matcher and @a timeslice match.
struct OutputRoute {
  size_t timeslice;
  size_t maxTimeslices;
  OutputSpec matcher;
  std::string channel;
};

} // namespace framework
} // namespace o2
#endif
