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

/// \file fdd-dcs-data-workflow.cxx
/// \brief Workflow for FDD DCS data processing
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "FDDDCSDataProcessorSpec.h"
#include "Framework/TypeTraits.h"

#include <unordered_map>

namespace o2::framework
{
template <>
struct has_root_dictionary<std::unordered_map<o2::dcs::DataPointIdentifier, o2::dcs::DataPointValue>, void> : std::true_type {
};
} // namespace o2::framework

#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  specs.emplace_back(getFDDDCSDataProcessorSpec());
  return specs;
}
