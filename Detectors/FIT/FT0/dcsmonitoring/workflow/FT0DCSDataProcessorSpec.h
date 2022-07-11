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

/// @file  FT0DCSDataProcessorSpec.h
/// @brief DataProcessorSpec for FT0 DCS data
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FT0_DATAPROCESSORSPEC_H
#define O2_FT0_DATAPROCESSORSPEC_H

#include "DetectorsCalibration/Utils.h"
#include "FT0DCSMonitoring/FT0DCSDataProcessor.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace framework
{

DataProcessorSpec getFT0DCSDataProcessorSpec()
{
  o2::header::DataDescription ddBChM = "FT0_DCSDPs";
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, ddBChM}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, ddBChM}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "ft0-dcs-data-processor",
    Inputs{{"input", "DCS", "FT0DATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::ft0::FT0DCSDataProcessor>("FT0", ddBChM)},
    Options{{"ccdb-path", VariantType::String, "http://localhost:8080", {"Path to CCDB"}},
            {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
            {"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}},
            {"DPs-update-interval", VariantType::Int64, 600ll, {"Interval (in s) after which to update the DPs CCDB entry"}}}};
}

} // namespace framework
} // namespace o2

#endif // O2_FT0_DATAPROCESSORSPEC_H
