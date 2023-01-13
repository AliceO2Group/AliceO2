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

/// \file PIDBase.cxx
/// \author Felix Schlepper

#include "TRDPID/PIDBase.h"
#include "DataFormatsTRD/PID.h"
#include "TRDPID/ML.h"
#include "TRDPID/Dummy.h"
#include "Framework/Logger.h"
#include "fmt/format.h"

namespace o2
{
namespace trd
{

std::unique_ptr<PIDBase> getTRDPIDBase(PIDPolicy policy)
{
  auto policyInt = static_cast<unsigned int>(policy);
  LOG(info) << "Creating PID policy. Loading model " << PIDPolicyEnum[policyInt];
  switch (policy) {
    case PIDPolicy::Test:
      return std::make_unique<XGB>(PIDPolicy::Test);
    case PIDPolicy::Dummy:
      return std::make_unique<Dummy>(PIDPolicy::Dummy);
    default:
      throw std::invalid_argument(fmt::format("Cannot create this PID policy {}({})", PIDPolicyEnum[policyInt], policyInt));
  }
  return nullptr; // cannot be reached
}

} // namespace trd
} // namespace o2
