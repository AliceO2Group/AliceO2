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

/// @file FDDDCSDataProcessorSpec.h
/// @brief DataProcessorSpec for FDD DCS data
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FDD_DATAPROCESSORSPEC_H
#define O2_FDD_DATAPROCESSORSPEC_H

#include "DetectorsCalibration/Utils.h"
#include "FDDDCSMonitoring/FDDDCSDataProcessor.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"

namespace o2
{
namespace framework
{

DataProcessorSpec getFDDDCSDataProcessorSpec()
{
  o2::header::DataDescription ddBChM = "FDD_DCSDPs";
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, ddBChM}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, ddBChM}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "fdd-dcs-data-processor",
    Inputs{{"input", "DCS", "FDDDATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::fdd::FDDDCSDataProcessor>("FDD", ddBChM)},
    Options{{"ccdb-path", VariantType::String, "http://localhost:8080", {"Path to CCDB"}},
            {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
            {"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}},
            {"DPs-update-interval", VariantType::Int64, 600ll, {"Interval (in s) after which to update the DPs CCDB entry"}}}};
}

} // namespace framework
} // namespace o2

#endif // O2_FDD_DATAPROCESSORSPEC_H
