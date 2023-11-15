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

/// @file FV0DCSDataProcessorSpec.h
/// @brief DataProcessorSpec for FV0 DCS data
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FV0_DATAPROCESSORSPEC_H
#define O2_FV0_DATAPROCESSORSPEC_H

#include "DetectorsCalibration/Utils.h"
#include "FV0DCSMonitoring/FV0DCSDataProcessor.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace framework
{

DataProcessorSpec getFV0DCSDataProcessorSpec()
{
  o2::header::DataDescription ddDCSDPs = "FV0_DCSDPs";
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, ddDCSDPs}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, ddDCSDPs}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "fv0-dcs-data-processor",
    Inputs{{"input", "DCS", "FV0DATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::fv0::FV0DCSDataProcessor>("FV0", ddDCSDPs)},
    Options{{"ccdb-path", VariantType::String, "http://localhost:8080", {"Path to CCDB"}},
            {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
            {"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}},
            {"DPs-update-interval", VariantType::Int64, 600ll, {"Interval (in s) after which to update the DPs CCDB entry"}}}};
}

} // namespace framework
} // namespace o2

#endif // O2_FV0_DATAPROCESSORSPEC_H
