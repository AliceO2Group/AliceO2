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

/// \file FDDDCSConfigProcessorSpec.cxx
/// \brief FDD processor spec for DCS configurations
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FDD_DCSCONFIGPROCESSOR_H
#define O2_FDD_DCSCONFIGPROCESSOR_H

#include "FITDCSMonitoring/FITDCSConfigProcessorSpec.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/WorkflowSpec.h"
#include "Headers/DataHeader.h"

#include <string>
#include <vector>

namespace o2
{

namespace framework
{

DataProcessorSpec getFDDDCSConfigProcessorSpec()
{
  o2::header::DataDescription ddDChM = "FDD_DCHM";
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, ddDChM}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, ddDChM}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "fdd-dcs-config-processor",
    Inputs{{"inputConfig", o2::header::gDataOriginFDD, "DCS_CONFIG_FILE", Lifetime::Timeframe},
           {"inputConfigFileName", o2::header::gDataOriginFDD, "DCS_CONFIG_NAME", Lifetime::Timeframe}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::fit::FITDCSConfigProcessor>("FDD", ddDChM)},
    Options{{"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}},
            {"filename-dchm", VariantType::String, "FDD-deadchannels.txt", {"Dead channel map file name"}},
            {"valid-days-dchm", VariantType::UInt32, 180u, {"Dead channel map validity in days"}},
            {"no-validate", VariantType::Bool, false, {"Don't validate the CCDB uploads"}}}};
}

} // namespace framework
} // namespace o2

#endif // O2_FDD_DCSCONFIGPROCESSOR_H
