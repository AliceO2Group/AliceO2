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

/// \file FT0DCSConfigProcessorSpec.cxx
/// \brief FT0 processor spec for DCS configurations
///
/// \author Andreas Molander <andreas.molander@cern.ch>, University of Jyvaskyla, Finland

#ifndef O2_FT0_DCSCONFIGPROCESSOR_SPEC_H
#define O2_FT0_DCSCONFIGPROCESSOR_SPEC_H

#include "FT0DCSMonitoring/FT0DCSConfigProcessor.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/WorkflowSpec.h"
#include "Headers/DataHeader.h"

#include <string>
#include <vector>

namespace o2::framework
{
DataProcessorSpec getFT0DCSConfigProcessorSpec()
{
  o2::header::DataDescription ddDChM = "FT0_DCHM";
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, ddDChM}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, ddDChM}, Lifetime::Sporadic);

  o2::header::DataDescription ddFeeConfig = "FT0_FEE_CONFIG";
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, ddFeeConfig}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, ddFeeConfig}, Lifetime::Sporadic);

  o2::header::DataDescription ddHvConfig = "FT0_HV_CONFIG";
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, ddHvConfig}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, ddHvConfig}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "ft0-dcs-config-processor",
    Inputs{{"inputConfig", o2::header::gDataOriginFT0, "DCS_CONFIG_FILE", Lifetime::Sporadic},
           {"inputConfigFileName", o2::header::gDataOriginFT0, "DCS_CONFIG_NAME", Lifetime::Sporadic}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::ft0::FT0DCSConfigProcessor>("FT0", ddDChM, ddFeeConfig, ddHvConfig)},
    Options{{"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}},
            {"filename-dchm", VariantType::String, "FT0-deadchannels.txt", {"Dead channel map file name"}},
            {"valid-days-dchm", VariantType::UInt32, 180u, {"Dead channel map validity in days"}},
            {"no-validate", VariantType::Bool, false, {"Don't validate the CCDB uploads"}},
            {"filename-fee-config", VariantType::String, "FT0-fee-config.json", {"FEE configuration file name"}},
            {"valid-days-fee-config", VariantType::UInt32, 180u, {"FEE configuration validity in days"}},
            {"filename-hv-config", VariantType::String, "FT0-hv-config.json", {"HV configuration file name"}},
            {"valid-days-hv-config", VariantType::UInt32, 180u, {"HV configuration validity in days"}}}};
}

} // namespace o2::framework

#endif // O2_FT0_DCSCONFIGPROCESSOR_SPEC_H
