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

#include "MCHBase/SanityCheck.h"
#include <fmt/core.h>

namespace o2::mch
{

bool isOK(const SanityError& error)
{
  return error.nofDuplicatedIndices == 0 &&
         error.nofDuplicatedItems == 0 &&
         error.nofMissingItems == 0 &&
         error.nofOutOfBounds == 0;
}

std::string asString(const SanityError& error)
{
  return fmt::format("error counts : {} duplicated items {} missing items {} out-of-bounds index {} duplicated index", error.nofDuplicatedItems, error.nofMissingItems, error.nofOutOfBounds, error.nofDuplicatedIndices);
}

} // namespace o2::mch
