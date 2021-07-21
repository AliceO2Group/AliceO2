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

#include "CommonDriverServices.h"
#include "Framework/CommonServices.h"

// Make sure we can use aggregated initialisers.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

namespace o2::framework
{

std::vector<ServiceSpec> o2::framework::CommonDriverServices::defaultServices()
{
  return {
    CommonServices::configurationSpec()};
}
} // namespace o2::framework

#pragma GCC diagnostic pop
